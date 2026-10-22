"""Structural and integrity checks for the exact assignment's output format."""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import Counter,defaultdict,deque
from pathlib import Path
from .simulate import COLUMNS, START_TS,DAY_S


def read_csv(path,name):
    with path.open(encoding='utf-8',newline='') as f:
        rd=csv.DictReader(f)
        if tuple(rd.fieldnames or []) != COLUMNS[name]:
            raise ValueError(f'{name}.csv: unexpected header: {rd.fieldnames}')
        return list(rd)


def check(data_dir,truth_dir,days=21):
    data_dir=Path(data_dir); truth_dir=Path(truth_dir)
    site=json.loads((data_dir/'site.json').read_text(encoding='utf-8'))
    assert site=={'tz':'Asia/Yekaterinburg','shifts':['08:00','20:00']}
    table={name:read_csv(data_dir/f'{name}.csv',name) for name in COLUMNS if name!='edge_events'}
    nodes,edges=table['nodes'],table['edges']
    assert len(nodes)>=60 and len([n for n in nodes if n['kind']=='load'])==3
    assert len([n for n in nodes if n['kind']=='dump'])==2
    assert all(n['kind'] in ('junction','load','dump') for n in nodes)
    point={n['node_id']:n for n in nodes}
    assert len(point)==len(nodes)
    assert len({e['edge_id'] for e in edges})==len(edges)
    assert set(e['surface'] for e in edges)=={'paved','gravel','dirt'}
    graph=defaultdict(list)
    for e in edges:
        u,v=e['from_node'],e['to_node']
        assert u in point and v in point and float(e['length_m'])>0
        dz=float(point[v]['z_m'])-float(point[u]['z_m'])
        assert abs(dz)/float(e['length_m'])<=.080001,(u,v,dz)
        graph[u].append(v)
    # Strong connectivity, not just an undirected component.
    def reach(start,graph):
        seen={start}; q=deque([start])
        while q:
            for v in graph[q.popleft()]:
                if v not in seen:
                    seen.add(v); q.append(v)
        return seen
    source=nodes[0]['node_id']
    assert len(reach(source,graph))==len(nodes)
    reversed_graph=defaultdict(list)
    for u,vals in graph.items():
        for v in vals:
            reversed_graph[v].append(u)
    assert len(reach(source,reversed_graph))==len(nodes)
    assert len(table['vehicles'])>=30
    assert len({v['model'] for v in table['vehicles']})>=2
    assert len({float(v['capacity_t']) for v in table['vehicles']})>=2
    weather=table['weather']
    assert len(weather)==days*24
    assert all(int(w['ts'])==START_TS+i*3600 for i,w in enumerate(weather))
    trips=table['trips']
    trip_ids={t['trip_id'] for t in trips}
    assert len(trip_ids)==len(trips)
    veh_ids={v['vehicle_id'] for v in table['vehicles']}
    assert all(t['vehicle_id'] in veh_ids and t['origin_node'] in point
               and t['dest_node'] in point for t in trips)
    assert all(START_TS<=int(t['depart_ts'])<int(t['arrive_ts'])<START_TS+days*DAY_S for t in trips)
    assert all(t['loaded'] in ('0','1') for t in trips)
    if days>=21:
        drivers={t['driver_id'] for t in trips}
        assert len(drivers)>=60, len(drivers)
        latest={t['driver_id'] for t in trips if int(t['depart_ts'])>=START_TS+(days-3)*DAY_S}
        earlier={t['driver_id'] for t in trips if int(t['depart_ts'])<START_TS+(days-3)*DAY_S}
        assert len(latest-earlier)>=5,latest-earlier
    edge_ids={e['edge_id'] for e in edges}
    event_count=0
    with (data_dir/'edge_events.csv').open(encoding='utf-8',newline='') as f:
        rows=csv.DictReader(f)
        assert tuple(rows.fieldnames or [])==COLUMNS['edge_events']
        for ev in rows:
            assert ev['trip_id'] in trip_ids and ev['edge_id'] in edge_ids
            assert int(ev['seq'])>=0 and int(ev['exit_ts'])>int(ev['enter_ts'])
            event_count+=1
    recs=table['recommendations']
    assert all(r['trip_id'] in trip_ids and r['accepted'] in ('0','1')
               and r['planned_node'] in point and r['recommended_node'] in point
               and r['planned_node']!=r['recommended_node'] for r in recs)
    trips_by_id={t['trip_id']:t for t in trips}
    for r in recs:
        trip=trips_by_id[r['trip_id']]
        assert trip['loaded']=='0' and trip['vehicle_id']==r['vehicle_id']
        chosen=r['recommended_node'] if r['accepted']=='1' else r['planned_node']
        assert trip['dest_node']==chosen
        assert math.isfinite(float(r['outcome_s'])) and float(r['outcome_s']) >= 0
        assert int(r['ts']) <= int(trip['depart_ts'])
    truth=json.loads((truth_dir/'effect.json').read_text(encoding='utf-8'))
    assert set(truth)=={'att_s','n_accepted'}
    assert truth['n_accepted']==sum(int(r['accepted']) for r in recs)
    print(f'VALID: {len(nodes)} nodes, {len(edges)} directed edges, {len(trips)} trips, '
          f'{event_count} edge events, {len(recs)} recommendations')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data'); p.add_argument('--truth',default='truth')
    p.add_argument('--days',type=int,default=21)
    args=p.parse_args()
    check(args.data,args.truth,args.days)

if __name__=='__main__':
    main()
