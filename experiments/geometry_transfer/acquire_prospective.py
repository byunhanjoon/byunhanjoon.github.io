#!/usr/bin/env python3
"""Acquire and preprocess the prospectively frozen four-source panel."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import requests


HERE=Path(__file__).resolve().parent
OUT=HERE/"prospective_data"
CONFIG=json.loads((HERE/"PROSPECTIVE_CONFIG.json").read_text())


def get(url: str, timeout=180) -> bytes:
    response=requests.get(url,timeout=timeout,headers={"User-Agent":"geometry-transfer-research/1.0"})
    response.raise_for_status(); return response.content


def haversine(latlon: np.ndarray) -> np.ndarray:
    x=np.radians(np.asarray(latlon,float));lat=x[:,0];lon=x[:,1]
    dlat=lat[:,None]-lat[None,:];dlon=lon[:,None]-lon[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 6371.0088*2*np.arcsin(np.sqrt(np.clip(a,0,1)))


def save_source(name: str, rows: pd.DataFrame, states: pd.DataFrame, distance: np.ndarray, domain_distance=None, metadata=None) -> dict:
    folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
    rows.to_parquet(folder/"rows.parquet",index=False);states.to_csv(folder/"states.csv",index=False)
    np.save(folder/"distance_primary.npy",np.asarray(distance,float))
    if domain_distance is not None: np.save(folder/"distance_domain.npy",np.asarray(domain_distance,float))
    payload={"status":"RUN","source":name,"rows":len(rows),"states":len(states),**(metadata or {})}
    (folder/"manifest.json").write_text(json.dumps(payload,indent=2)+"\n");return payload


def ghcn() -> dict:
    cfg=CONFIG["sources"]["noaa_ghcn_tmax"]
    stations_text=get(cfg["metadata_url"]).decode("ascii","ignore")
    inventory=get(cfg["inventory_url"]).decode("ascii","ignore")
    meta={}
    for line in stations_text.splitlines():
        try:
            sid=line[:11].strip();lat=float(line[12:20]);lon=float(line[21:30]);elev=float(line[31:37])
            meta[sid]=(lat,lon,elev,line[41:71].strip())
        except ValueError: continue
    candidates=[]
    for line in inventory.splitlines():
        if len(line)<45: continue
        sid=line[:11].strip();element=line[31:35];first=line[36:40];last=line[41:45]
        if sid.startswith("US") and element=="TMAX" and first<="2024"<=last and sid in meta:
            lat,lon,_,_=meta[sid]
            if 25<=lat<=49.5 and -125<=lon<=-66:
                key=hashlib.sha256(f"ghcn|{sid}".encode()).hexdigest();candidates.append((key,sid))
    selected=[];records=[]
    for _,sid in sorted(candidates):
        if len(selected)>=cfg["max_states"]: break
        try: lines=get(cfg["station_url_template"].format(station=sid),timeout=60).decode("ascii","ignore").splitlines()
        except Exception: continue
        station_rows=[]
        for line in lines:
            if len(line)<269 or line[11:15]!="2024" or line[17:21]!="TMAX": continue
            month=int(line[15:17])
            for day in range(1,32):
                offset=21+(day-1)*8
                try: value=int(line[offset:offset+5])
                except ValueError: continue
                if value==-9999: continue
                try: date=pd.Timestamp(2024,month,day)
                except ValueError: continue
                station_rows.append((sid,date,value/10.0))
        if len(station_rows)>=300:
            selected.append(sid);records.extend(station_rows)
    rows=pd.DataFrame(records,columns=["field_state","date","target"])
    rows["day_of_year"]=rows.date.dt.dayofyear;rows["month"]=rows.date.dt.month
    rows["sin_doy"]=np.sin(2*np.pi*rows.day_of_year/366);rows["cos_doy"]=np.cos(2*np.pi*rows.day_of_year/366)
    states=pd.DataFrame([{"state_id":s,"latitude":meta[s][0],"longitude":meta[s][1],"elevation":meta[s][2],"name":meta[s][3]} for s in selected])
    elev=dict(zip(states.state_id,states.elevation));rows["elevation"]=rows.field_state.map(elev)
    return save_source("noaa_ghcn_tmax",rows.drop(columns="date"),states,haversine(states[["latitude","longitude"]].to_numpy()),metadata={"ordinary_covariates":["day_of_year","month","sin_doy","cos_doy","elevation"],"family":"NOAA_GHCN"})


def beijing() -> dict:
    cfg=CONFIG["sources"]["beijing_pm25"];outer=zipfile.ZipFile(io.BytesIO(get(cfg["url"])))
    nested=next(n for n in outer.namelist() if n.lower().endswith(".zip"))
    z=zipfile.ZipFile(io.BytesIO(outer.read(nested)))
    coords={"Aotizhongxin":(39.982,116.397),"Changping":(40.218,116.230),"Dingling":(40.292,116.220),"Dongsi":(39.929,116.417),"Guanyuan":(39.929,116.339),"Gucheng":(39.914,116.184),"Huairou":(40.328,116.628),"Nongzhanguan":(39.937,116.461),"Shunyi":(40.127,116.655),"Tiantan":(39.886,116.407),"Wanliu":(39.987,116.287),"Wanshouxigong":(39.878,116.352)}
    frames=[]
    for name in z.namelist():
        if not name.lower().endswith(".csv"): continue
        frame=pd.read_csv(z.open(name));station=str(frame["station"].iloc[0]);frame["field_state"]=station;frame["target"]=pd.to_numeric(frame["PM2.5"],errors="coerce");frames.append(frame)
    rows=pd.concat(frames,ignore_index=True).dropna(subset=["target"])
    rows["wd"]=rows.wd.astype("string").fillna("missing").astype(str)
    ordinary=["year","month","day","hour","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","wd","WSPM"]
    states=pd.DataFrame([{"state_id":s,"latitude":coords[s][0],"longitude":coords[s][1]} for s in sorted(rows.field_state.unique())])
    return save_source("beijing_pm25",rows[["field_state","target",*ordinary]],states,haversine(states[["latitude","longitude"]].to_numpy()),metadata={"ordinary_covariates":ordinary,"family":"UCI_BEIJING_AIR"})


def divvy() -> dict:
    cfg=CONFIG["sources"]["divvy_trip_duration"];z=zipfile.ZipFile(io.BytesIO(get(cfg["url"])))
    csv=next(n for n in z.namelist() if n.lower().endswith(".csv"));frame=pd.read_csv(z.open(csv),low_memory=False)
    # Normalize the frozen Q1 schema.
    aliases={"start_station_id":"start_station_id","end_station_id":"end_station_id","start_lat":"start_lat","start_lng":"start_lng","ended_at":"ended_at","started_at":"started_at"}
    if "starttime" in frame.columns:
        aliases={"start_station_id":"from_station_id","end_station_id":"to_station_id","start_lat":"start_lat","start_lng":"start_lng","ended_at":"stoptime","started_at":"starttime"}
    start_id=aliases["start_station_id"];end_id=aliases["end_station_id"]
    frame["started_at"]=pd.to_datetime(frame[aliases["started_at"]]);frame["ended_at"]=pd.to_datetime(frame[aliases["ended_at"]])
    frame["seconds"]=(frame.ended_at-frame.started_at).dt.total_seconds()
    frame=frame[(frame.seconds>=60)&(frame.seconds<=14400)].dropna(subset=[start_id,end_id,aliases["start_lat"],aliases["start_lng"]])
    counts=frame[start_id].astype(str).value_counts();selected=counts[counts>=50].head(80).index.astype(str)
    frame["field_state"]=frame[start_id].astype(str);frame["end_station"]=frame[end_id].astype(str)
    frame=frame[frame.field_state.isin(selected)].copy();frame["target"]=np.log1p(frame.seconds)
    frame["hour"]=frame.started_at.dt.hour;frame["day_of_week"]=frame.started_at.dt.dayofweek
    member="member_casual" if "member_casual" in frame else ("usertype" if "usertype" in frame else None)
    ride="rideable_type" if "rideable_type" in frame else None
    if member is None: frame["member_type"]="unknown"
    else: frame["member_type"]=frame[member].astype(str)
    if ride is None: frame["ride_type"]="unknown"
    else: frame["ride_type"]=frame[ride].astype(str)
    states=[]
    for s in selected:
        g=frame[frame.field_state==s]
        states.append({"state_id":s,"latitude":float(g[aliases["start_lat"]].median()),"longitude":float(g[aliases["start_lng"]].median())})
    states=pd.DataFrame(states);geo=haversine(states[["latitude","longitude"]].to_numpy());lookup={s:i for i,s in enumerate(states.state_id)}
    graph=nx.Graph();graph.add_nodes_from(range(len(states)))
    for a,b in frame[["field_state","end_station"]].drop_duplicates().itertuples(index=False):
        if a in lookup and b in lookup:
            i,j=lookup[a],lookup[b];graph.add_edge(i,j,weight=max(geo[i,j],.05))
    domain=np.full_like(geo,np.inf)
    for i,lengths in nx.all_pairs_dijkstra_path_length(graph,weight="weight"):
        for j,value in lengths.items():domain[i,j]=value
    disconnected=~np.isfinite(domain);domain[disconnected]=geo[disconnected]+2*float(np.nanmax(geo));np.fill_diagonal(domain,0)
    ordinary=["hour","day_of_week","member_type","ride_type","end_station"]
    return save_source("divvy_trip_duration",frame[["field_state","target",*ordinary]],states,geo,domain,metadata={"ordinary_covariates":ordinary,"family":"CHICAGO_DIVVY"})


def bls() -> dict:
    cfg=CONFIG["sources"]["bls_oews_wage"];z=zipfile.ZipFile(io.BytesIO(get(cfg["url"])))
    name=next(n for n in z.namelist() if n.lower().endswith((".xlsx",".xls")))
    frame=pd.read_excel(io.BytesIO(z.read(name)));frame.columns=[str(c).upper() for c in frame.columns]
    group="O_GROUP" if "O_GROUP" in frame else None
    if group: frame=frame[frame[group].astype(str).str.lower().eq("detailed")]
    frame["target_raw"]=pd.to_numeric(frame["A_MEAN"],errors="coerce");frame=frame.dropna(subset=["target_raw","OCC_CODE","AREA"])
    frame=frame[frame.target_raw>0].copy();frame["field_state"]=frame.OCC_CODE.astype(str);frame["target"]=np.log(frame.target_raw)
    frame["employment"]=pd.to_numeric(frame.get("TOT_EMP"),errors="coerce")
    valid=frame.field_state.value_counts();valid=valid[valid>=10].index;frame=frame[frame.field_state.isin(valid)]
    ids=sorted(frame.field_state.unique());states=pd.DataFrame({"state_id":ids})
    paths=[]
    for code in ids:
        digits=re.sub(r"\D","",code);paths.append(["root",digits[:2],digits[:3],digits[:5],digits[:6]])
    d=np.zeros((len(ids),len(ids)))
    for i,left in enumerate(paths):
        for j,right in enumerate(paths):
            common=sum(a==b for a,b in zip(left,right));d[i,j]=2*(len(left)-common)
    return save_source("bls_oews_wage",frame[["field_state","target","AREA","employment"]].rename(columns={"AREA":"area"}),states,d,metadata={"ordinary_covariates":["area","employment"],"family":"BLS_OEWS"})


def main() -> None:
    OUT.mkdir(exist_ok=True);results=[]
    for name,fn in (("noaa_ghcn_tmax",ghcn),("beijing_pm25",beijing),("divvy_trip_duration",divvy),("bls_oews_wage",bls)):
        try: result=fn()
        except Exception as exc:
            result={"status":"NOT RUN — SOURCE UNAVAILABLE","source":name,"error":f"{type(exc).__name__}: {exc}"}
            folder=OUT/name;folder.mkdir(parents=True,exist_ok=True);(folder/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
        print(result,flush=True);results.append(result)
    (OUT/"acquisition_summary.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":main()
