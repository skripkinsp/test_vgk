"""Event-based quarry fleet simulator: no ML, no network and no third-party deps.

Only the columns defined in the assignment are written to data/*.csv.
All hidden simulation state stays in memory; simulator-only truth goes to truth/.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .graph import generate_graph, k_shortest_paths

COLUMNS = {
    'nodes': ('node_id','kind','x_m','y_m','z_m'),
    'edges': ('edge_id','from_node','to_node','length_m','surface'),
    'vehicles': ('vehicle_id','model','capacity_t'),
    'weather': ('ts','precip_mm_h','visibility_m'),
    'trips': ('trip_id','vehicle_id','driver_id','origin_node','dest_node','loaded','payload_t','depart_ts','arrive_ts'),
    'edge_events': ('trip_id','seq','edge_id','enter_ts','exit_ts'),
    'recommendations': ('rec_id','ts','vehicle_id','trip_id','planned_node','recommended_node','est_gain_s','accepted','outcome_s'),
}
MODELS = (
    {'model': 'Small', 'capacity_t': 45, 'empty_kmh': 32, 'loaded_kmh': 25},
    {'model': 'Medium', 'capacity_t': 70, 'empty_kmh': 30, 'loaded_kmh': 22},
    {'model': 'Heavy', 'capacity_t': 100, 'empty_kmh': 27, 'loaded_kmh': 19},
)
SURFACE_FACTOR = {'paved': 1.0, 'gravel': 0.83, 'dirt': 0.69}
DAY_S = 86_400
START_TS = int(datetime(2025, 9, 1, 3, tzinfo=timezone.utc).timestamp())  # 08:00 site time


def write_csv(path, schema, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(schema)
        for row in rows:
            writer.writerow([row[col] for col in schema])


def sigmoid(x):
    if x >= 0:
        return 1/(1+math.exp(-x))
    e = math.exp(x)
    return e/(1+e)


class Simulator:
    def __init__(self, seed=42, days=21, nodes=200, selection_strength=1.0, causal_effect_scale=1.0):
        if days < 1:
            raise ValueError('days must be >= 1')
        self.rng = random.Random(seed)
        self.telemetry_rng = random.Random(seed + 921_771)
        # Independent RNG: causal outcomes must not affect fleet trajectories.
        self.causal_rng = random.Random(seed + 6_028_471)
        self.days = days
        self.selection_strength = selection_strength
        self.causal_effect_scale = causal_effect_scale
        self.end_ts = START_TS+days*DAY_S
        self.nodes, self.edges, self.adj, self.loads, self.dumps = generate_graph(self.rng, nodes)
        self.vehicles = [dict(vehicle_id=f'v-{i:02d}', model=MODELS[i%3]['model'],
                              capacity_t=MODELS[i%3]['capacity_t']) for i in range(30)]
        self.drivers = {
            f'd-{i:02d}': {'speed': self.rng.uniform(.87, 1.13),
                            'caution': self.rng.betavariate(2, 2)} for i in range(66)
        }
        self.weather, self.wet = self.make_weather()
        self.route_cache = {}
        self.events = []
        self.serial = 0
        self.trip_rows, self.rec_rows = [], []
        # Completed trips' telemetry is streamed to disk, not kept in RAM.
        self.edge_stream = tempfile.TemporaryFile(mode='w+', encoding='utf-8', newline='')
        self.edge_writer = csv.writer(self.edge_stream,lineterminator='\n',quoting=csv.QUOTE_MINIMAL)
        self.edge_writer.writerow(COLUMNS['edge_events'])
        self.edge_count = 0
        self.trips_state = {}
        self.traffic = defaultdict(list)  # physical edge -> min-heap of planned exit timestamps
        self.load_free = {p: START_TS for p in self.loads}
        self.failure_budget = [self.rng.expovariate(1/(90*3600)) for _ in range(30)]
        self.breakdowns = 0
        self.stuck_events = 0
        self.congested_entries = 0
        self.last_driver = {}

    def make_weather(self):
        weather = []
        wetness = []
        wet = 0.
        rain = 0.
        storm_remaining = 0
        for h in range(self.days*24):
            if storm_remaining <= 0 and self.rng.random() < .13:
                storm_remaining = self.rng.randint(2, 7)
                rain = self.rng.uniform(5, 19)
            if storm_remaining > 0:
                rain = max(0, rain + self.rng.uniform(-3, 2.5))
                storm_remaining -= 1
            else:
                rain *= .38
            rain = round(rain if rain >= .15 else 0, 2)
            # 4-5 hour washout: remains wet after the rain stops.
            wet = .78*wet + .22*rain
            wetness.append(min(1., wet/10))
            weather.append({'ts': START_TS+h*3600, 'precip_mm_h': rain,
                            'visibility_m': round(max(300, 10000-rain*420+self.rng.uniform(-700, 700)))})
        return weather, wetness

    def weather_now(self, t):
        idx = max(0, min(len(self.weather)-1, int((t-START_TS)//3600)))
        return self.weather[idx], self.wet[idx]

    def local_hour_minute(self, t):
        # Asia/Yekaterinburg is UTC+05 without DST in the simulated period.
        seconds = (t+5*3600) % DAY_S
        return seconds//3600, (seconds % 3600)//60

    def driver_at(self, vehicle_idx, t):
        hr, _ = self.local_hour_minute(t)
        shift = 0 if 8 <= hr < 20 else 1
        # Six entirely unseen drivers replace six shifts on the final three days.
        if t >= self.end_ts-3*DAY_S and vehicle_idx < 6:
            return f'd-{60+vehicle_idx:02d}'
        return f'd-{vehicle_idx*2+shift:02d}'

    def norm_speed(self, edge, vehicle_idx, loaded):
        """Visible static nominal physics used for route candidates."""
        model = MODELS[vehicle_idx%3]
        kmh = model['loaded_kmh'] if loaded else model['empty_kmh']
        uphill = max(0, edge['slope'])
        slope_factor = 1/(1+(11 if loaded else 5)*uphill)
        downhill = max(0, -edge['slope'])
        downhill_factor = min(1.08, 1 + .7*downhill)
        return max(4., kmh*SURFACE_FACTOR[edge['surface']]*slope_factor*downhill_factor)

    def routes(self, start, dest, vehicle_idx, loaded):
        # Speeds are deterministic here; real rain, delays, traffic happen downstream.
        key = (start, dest, vehicle_idx%3, int(loaded))
        if key not in self.route_cache:
            cost = lambda e: e['length_m']/(self.norm_speed(e, vehicle_idx, loaded)/3.6)
            self.route_cache[key] = k_shortest_paths(self.adj, self.edges, start, dest, cost, k=5)
        return self.route_cache[key]

    def route_score(self, route, vehicle_idx, loaded, driver_id, t):
        _, wet = self.weather_now(t)
        base = sum(e['length_m']/(self.norm_speed(e, vehicle_idx, loaded)/3.6)
                   for idx in route for e in (self.edges[idx],))
        # A cautious driver tolerates a longer route to avoid wet dirt and steep ramps.
        risk = sum((8*wet if self.edges[idx]['surface']=='dirt' else
                    2*wet if self.edges[idx]['surface']=='gravel' else 0)
                   + 8*max(0,self.edges[idx]['slope'])/.08 for idx in route)
        return base + self.drivers[driver_id]['caution']*risk

    def route_for_driver(self, start, dest, vehicle_idx, loaded, driver_id, t):
        alternatives = self.routes(start, dest, vehicle_idx, loaded)
        if not alternatives:
            raise ValueError(f'No route from {start} to {dest}')
        scores = [self.route_score(r,vehicle_idx,loaded,driver_id,t) for r in alternatives]
        return min(range(len(alternatives)), key=lambda i: scores[i]+self.rng.gauss(0, 5)), alternatives

    def nominal_to_load(self, from_node, dest, vehicle_idx, driver_id, t):
        """Pre-decision snapshot, not a peek at the future, for recommendation policy.

        Important limitation: this does NOT simulate system-wide counterfactuals.
        """
        routes = self.routes(from_node,dest,vehicle_idx,False)
        if not routes:
            raise ValueError('no route to loading station')
        route = min(routes, key=lambda r: self.route_score(r,vehicle_idx,False,driver_id,t))
        _, wet = self.weather_now(t)
        factor = self.drivers[driver_id]['speed']
        travel = sum(e['length_m'] / (max(2, self.norm_speed(e,vehicle_idx,False)
                                      *factor*(1 - wet*(.30 if e['surface']=='dirt' else
                                                       .11 if e['surface']=='gravel' else .035)))/3.6)
                     for idx in route for e in (self.edges[idx],))
        wait = max(0, self.load_free[dest]-(t+travel))
        return travel+wait

    def push(self, t, kind, payload):
        self.serial += 1
        heapq.heappush(self.events, (int(t), self.serial, kind, payload))

    def reserve_load(self, vehicle_idx, load_node, t, rec_index=None):
        start_service = max(t, self.load_free[load_node])
        service_s = self.rng.randint(180, 360)
        end_service = start_service+service_s
        self.load_free[load_node] = end_service
        if rec_index is not None:
            rec = self.rec_rows[rec_index]
            # The public recommendation outcome is generated by a separate
            # pre-treatment structural causal model. It need not equal the
            # simulated operational start_service - rec['ts']; that quantity
            # remains relevant only to the physical fleet simulator.
            # Both potential outcomes were sampled BEFORE treatment assignment.
            rec['outcome_s'] = rec['_y1'] if rec['accepted'] else rec['_y0']
        dump = self.rng.choice(self.dumps)
        self.push(end_service, 'depart', (vehicle_idx,load_node,dump,1,None))

    def start_trip(self, vehicle_idx, start, dest, loaded, rec_index, t):
        driver = self.driver_at(vehicle_idx,t)
        previous = self.last_driver.get(vehicle_idx)
        if previous is not None and previous != driver:
            t += self.rng.randint(100,240)  # shift handover before driving
        self.last_driver[vehicle_idx] = driver
        route_idx, alternatives = self.route_for_driver(start,dest,vehicle_idx,loaded,driver,t)
        path = alternatives[route_idx]
        trip_id = f'tr-{len(self.trip_rows):07d}'
        mass = round(self.vehicles[vehicle_idx]['capacity_t']*self.rng.uniform(.78,.98),2) if loaded else 0.
        tr = dict(trip_id=trip_id, vehicle_id=self.vehicles[vehicle_idx]['vehicle_id'],
                  driver_id=driver, origin_node=self.nodes[start]['node_id'],
                  dest_node=self.nodes[dest]['node_id'], loaded=int(loaded), payload_t=mass,
                  depart_ts=t, arrive_ts=0)
        self.trip_rows.append(tr)
        self.trips_state[trip_id] = {'i': vehicle_idx, 'dest': dest, 'loaded': loaded,
                                     'driver':driver, 'route':path, 'step':0,
                                     'rec_index':rec_index, 'edge_events':[]}
        if rec_index is not None:
            self.rec_rows[rec_index]['trip_id'] = trip_id
        self.push(t,'edge',trip_id)

    def drive_next_edge(self, trip_id, t):
        st = self.trips_state[trip_id]
        i = st['i']; loaded = st['loaded']; driver = st['driver']
        route = st['route']; pos = st['step']
        if pos == len(route):
            self.trip_rows[int(trip_id[3:])]['arrive_ts'] = t
            self.flush_trip_events(st['edge_events'], self.vehicles[i]['vehicle_id'])
            if loaded:
                self.push(t+self.rng.randint(90,180),'recommend', (i,st['dest']))
            else:
                self.reserve_load(i,st['dest'],t,st['rec_index'])
            del self.trips_state[trip_id]
            return
        e = self.edges[route[pos]]
        occupancy = self.traffic[e['physical']]
        while occupancy and occupancy[0] <= t:
            heapq.heappop(occupancy)
        n_traffic = len(occupancy)
        self.congested_entries += int(n_traffic >= 1)
        weather, wet = self.weather_now(t)
        base_kmh = self.norm_speed(e,i,loaded)
        rain_factor = 1 - wet*({'dirt':.48, 'gravel':.17, 'paved':.04}[e['surface']])
        hr, minute = self.local_hour_minute(t)
        night = .88 if hr < 8 or hr >= 20 else 1.
        traffic_factor = 1/(1+.16*n_traffic)
        speed_kmh = max(5 if loaded else 8,
                        min(40,base_kmh*rain_factor*night*traffic_factor
                            *self.drivers[driver]['speed']*self.rng.uniform(.96,1.04)))
        driving_sec = max(1, math.ceil(e['length_m']/(speed_kmh/3.6)))
        stopped = 0
        # Failure clocks are operating time, not wall-clock time.
        self.failure_budget[i] -= driving_sec
        if self.failure_budget[i] <= 0:
            stopped += self.rng.randint(600,3600)
            self.breakdowns += 1
            self.failure_budget[i] += self.rng.expovariate(1/(90*3600))
        if e['surface']=='dirt' and self.rng.random() < .0025*wet**2*(1.8 if loaded else 1.):
            stopped += self.rng.randint(120,900)
            self.stuck_events += 1
        if hr in (8,20) and minute < 12 and self.rng.random() < .06:
            stopped += self.rng.randint(60,240)
        exit_ts = t+driving_sec+stopped
        heapq.heappush(occupancy,exit_ts)
        st['edge_events'].append({'trip_id':trip_id,'seq':pos,'edge_id':e['edge_id'],
                                  'enter_ts':t,'exit_ts':exit_ts})
        st['step'] += 1
        self.push(exit_ts,'edge',trip_id)

    def make_recommendation(self, vehicle_idx, dump, t):
        driver = self.driver_at(vehicle_idx,t)
        planned = self.rng.choice(self.loads)
        alternatives = [n for n in self.loads if n != planned]
        nominal_plan = self.nominal_to_load(dump, planned, vehicle_idx, driver, t)
        potential = [(self.nominal_to_load(dump, n, vehicle_idx, driver, t),n) for n in alternatives]
        nominal_alt, suggested = min(potential)
        # Biased noisy advice; some recommendations may not help.
        true_snapshot_gain = nominal_plan - nominal_alt
        est_gain = true_snapshot_gain + 60 + self.rng.gauss(0,75)
        hour,_ = self.local_hour_minute(t)
        # Observable-confounding scenario: treatment assignment uses ONLY
        # pre-decision information published in recommendations.csv/vehicles.csv.
        # Never condition acceptance on hidden queue state or true_snapshot_gain.
        # Strength=0 corresponds to randomized acceptance (constant propensity).
        model = self.vehicles[vehicle_idx]['model']
        model_bonus = {'Small': -0.15, 'Medium': 0.0, 'Heavy': 0.15}[model]
        night_bonus = 0.25 if hour < 8 or hour >= 20 else 0.0
        score = (-0.6 + self.selection_strength * (
            0.003 * (est_gain - 150.0) + night_bonus + model_bonus))
        # Pre-treatment structural outcome model (separate from the fleet ETA).
        # All systematic causes are either public features or independent
        # exogenous noise. This makes exchangeability conditional on public X
        # testable and avoids post-treatment residual counterfactuals.
        # The recommendation effect is heterogeneous, but does not depend on
        # which branch will eventually be selected.
        rain, _ = self.weather_now(t)
        baseline = (610.0 + 0.32*est_gain
                    + {'Small': -35.0, 'Medium': 0.0, 'Heavy': 45.0}[model]
                    + (35.0 if night_bonus else 0.0)
                    + 1.8*rain['precip_mm_h'])
        shared_noise = self.causal_rng.gauss(0, 55)
        individual_saving = max(-80.0, min(450.0,
            100.0 + 0.48*est_gain + self.causal_rng.gauss(0, 25)))
        y0 = max(60.0, baseline + shared_noise)
        y1 = max(60.0, y0 - self.causal_effect_scale * individual_saving)
        accepted = int(self.rng.random() < max(.08, min(.92, sigmoid(score))))
        rec_id = f'rec-{len(self.rec_rows):07d}'
        self.rec_rows.append({'rec_id':rec_id,'ts':t,
             'vehicle_id':self.vehicles[vehicle_idx]['vehicle_id'],'trip_id':'',
             'planned_node':self.nodes[planned]['node_id'],
             'recommended_node':self.nodes[suggested]['node_id'],
             'est_gain_s':round(est_gain,2),'accepted':accepted,'outcome_s':-1,
             '_nominal_y0':float(nominal_plan),
             '_nominal_y1':float(nominal_alt),
             '_y0':float(y0), '_y1':float(y1),
             '_residual':float(shared_noise)})
        chosen = suggested if accepted else planned
        self.push(t,'depart',(vehicle_idx,dump,chosen,0,len(self.rec_rows)-1))

    def run(self):
        for i in range(30):
            self.push(START_TS+self.rng.randrange(90),'initial_load',(i,self.loads[i%3]))
        while self.events:
            t,_,kind,data = heapq.heappop(self.events)
            if t >= self.end_ts:
                break
            if kind=='initial_load':
                i,load_node = data
                self.reserve_load(i,load_node,t)
            elif kind=='depart':
                i,start,dest,loaded,rec_idx = data
                self.start_trip(i,start,dest,loaded,rec_idx,t)
            elif kind=='edge':
                self.drive_next_edge(data,t)
            elif kind=='recommend':
                i,dump = data
                self.make_recommendation(i,dump,t)
            else:
                raise RuntimeError(f'unknown event {kind}')
        completed = {tr['trip_id'] for tr in self.trip_rows if 0 < tr['arrive_ts'] < self.end_ts}
        self.trip_rows = [tr for tr in self.trip_rows if tr['trip_id'] in completed]
        self.rec_rows = [rec for rec in self.rec_rows if rec['trip_id'] in completed
                         and rec['outcome_s'] >= 0 and rec['ts']+rec['outcome_s'] < self.end_ts]
        # Only completed trips with observed causal outcomes contribute.
        # Consistency holds by construction: Y(accepted) == outcome_s.
        for rec in self.rec_rows:
            assert abs(rec['outcome_s'] - rec['_y1' if rec['accepted'] else '_y0']) < 1e-9
        accepted = [rec['_y1'] - rec['_y0'] for rec in self.rec_rows if rec['accepted']]
        truth = {'att_s':round(sum(accepted)/len(accepted),3) if accepted else 0.0,
                 'n_accepted':len(accepted)}
        return truth

    def flush_trip_events(self, events, vehicle_id):
        """Streaming telemetry corruption; gaps in seq are intentional."""
        rng = self.telemetry_rng
        for ev in events:
            if rng.random() < .03:
                continue
            ts0, ts1 = ev['enter_ts'], ev['exit_ts']
            if (vehicle_id in ('v-00','v-01') and
                    START_TS+7*DAY_S <= ts0 < START_TS+13*DAY_S):
                ts0 += 180
                ts1 += 180
            row = (ev['trip_id'],ev['seq'],ev['edge_id'],ts0,ts1)
            self.edge_writer.writerow(row)
            self.edge_count += 1
            if rng.random() < .005:
                self.edge_writer.writerow(row)
                self.edge_count += 1

    def save(self, data_dir, truth_dir, truth):
        data_dir, truth_dir = Path(data_dir), Path(truth_dir)
        data_dir.mkdir(parents=True,exist_ok=True)
        truth_dir.mkdir(parents=True,exist_ok=True)
        (data_dir/'site.json').write_text(json.dumps({'tz':'Asia/Yekaterinburg',
                   'shifts':['08:00','20:00']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        data = {'nodes':self.nodes,'edges':self.edges,'vehicles':self.vehicles,
                'weather':self.weather,'trips':self.trip_rows,
                'recommendations':self.rec_rows}
        for name,rows in data.items():
            write_csv(data_dir/f'{name}.csv',COLUMNS[name],rows)
        self.edge_stream.flush()
        self.edge_stream.seek(0)
        with (data_dir/'edge_events.csv').open('w',encoding='utf-8',newline='') as f:
            shutil.copyfileobj(self.edge_stream,f)
        self.edge_stream.close()
        (truth_dir/'effect.json').write_text(json.dumps(truth,indent=2)+'\n',encoding='utf-8')
        # Private diagnostic file, deliberately outside DATA_DIR. Never use it
        # to fit propensity/outcome models or construct inference features.
        potential_columns = ('rec_id', 'accepted', 'outcome_s', 'y0_s', 'y1_s',
                             'individual_effect_s', 'est_gain_s', 'shared_residual_s')
        potentials = ({'rec_id':r['rec_id'], 'accepted':r['accepted'],
                       'outcome_s':r['outcome_s'], 'y0_s':round(r['_y0'],6),
                       'y1_s':round(r['_y1'],6),
                       'individual_effect_s':round(r['_y1']-r['_y0'],6),
                       'est_gain_s':r['est_gain_s'],
                       'shared_residual_s':round(r['_residual'],6)}
                      for r in self.rec_rows)
        write_csv(truth_dir/'potential_outcomes.csv', potential_columns, potentials)
        print(json.dumps({'nodes':len(self.nodes),'edges':len(self.edges),
            'trips':len(self.trip_rows),'edge_events':self.edge_count,
            'recommendations':len(self.rec_rows),'accepted':truth['n_accepted'],
            'structural_att_s':truth['att_s'],'breakdowns':self.breakdowns,
            'stuck_events':self.stuck_events,'congested_entries':self.congested_entries},indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='data')
    p.add_argument('--truth',default='truth')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--days',type=int,default=21)
    p.add_argument('--nodes',type=int,default=200)
    p.add_argument('--selection-strength',type=float,default=1.)
    p.add_argument('--causal-effect-scale',type=float,default=1.,
                   help='0 creates a zero-effect causal validation scenario')
    a=p.parse_args()
    sim=Simulator(a.seed,a.days,a.nodes,a.selection_strength,a.causal_effect_scale)
    truth=sim.run()
    sim.save(a.out,a.truth,truth)

if __name__=='__main__':
    main()
