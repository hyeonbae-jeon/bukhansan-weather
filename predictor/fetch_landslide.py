"""
산림청 디지털사면통합 산사태정보시스템(sansatai.forest.go.kr) API에서
'산사태 취약지역 관리대장(weakRegisterList)'을 받아와, 북한산 국립공원 경계 안에
있는 지점만 걸러서 GeoJSON으로 저장한다.

이 API는 위치 기반 검색 파라미터가 없고 전국 데이터를 페이지 단위로만 내려주므로,
전체를 순회하며 받은 뒤 우리 쪽에서 경계로 필터링한다 (국가 전체 데이터라 페이지 수가
꽤 될 수 있음 - 초당 요청 제한을 감안해 약간의 딜레이를 둔다).

사용법:
    export SANSATAI_API_KEY="발급받은 인증키"
    python fetch_landslide.py --boundary ../frontend/geo/boundary.geojson --out ../frontend/geo/landslide.geojson

주의:
- 공식 응답 스펙 문서에 '결과를 감싸는 최상위 키'가 명시되어 있지 않아서, 흔한 패턴
  (list/items/data 또는 배열 그 자체)을 모두 시도하도록 방어적으로 짰다. 만약 실행했을 때
  0건으로 나오면, 응답 원본 JSON 구조가 다른 것이니 --debug 옵션으로 원본을 출력해서 보여달라.
"""
import argparse
import json
import os
import sys
import time

import requests

BASE = "http://sansatai.forest.go.kr/lsapi/openapi/weakRegisterList.json"


def extract_rows(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("list", "items", "data", "result", "resultList", "body"):
            v = data.get(key)
            if isinstance(v, list):
                return v
            if isinstance(v, dict):
                inner = extract_rows(v)
                if inner:
                    return inner
    return []


def fetch_all(api_key: str, max_pages: int = 300, page_size: int = 500, debug: bool = False):
    all_rows = []
    page = 1
    while page <= max_pages:
        params = {"apikey": api_key, "pageno": page, "result": page_size}
        resp = requests.get(BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if debug and page == 1:
            print("[디버그] 1페이지 원본 응답(앞부분):", json.dumps(data, ensure_ascii=False)[:1000], file=sys.stderr)

        rows = extract_rows(data)
        if not rows:
            print(f"[page {page}] 더 이상 데이터 없음, 종료")
            break
        all_rows.extend(rows)
        print(f"[page {page}] {len(rows)}건 (누적 {len(all_rows)}건)")
        if len(rows) < page_size:
            break
        page += 1
        time.sleep(0.15)
    return all_rows


def point_in_ring(lat, lon, ring):
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def point_in_boundary(lat, lon, boundary_geo):
    geom = boundary_geo["features"][0]["geometry"]
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    return any(point_in_ring(lat, lon, poly[0]) for poly in polys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("SANSATAI_API_KEY")
    if not api_key:
        print("환경변수 SANSATAI_API_KEY가 설정되어 있지 않습니다.", file=sys.stderr)
        sys.exit(1)

    with open(args.boundary, encoding="utf-8") as f:
        boundary_geo = json.load(f)

    rows = fetch_all(api_key, debug=args.debug)
    print(f"전국 취약지역 총 {len(rows)}건 수신")

    features = []
    skipped_no_coord = 0
    for r in rows:
        lat_raw = r.get("vnaraLctnLttdVal")
        lon_raw = r.get("vnaraLctnLngtdVal")
        try:
            lat = float(lat_raw)
            lon = float(lon_raw)
        except (TypeError, ValueError):
            skipped_no_coord += 1
            continue
        if not point_in_boundary(lat, lon, boundary_geo):
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "address": r.get("vnaraExmnnArcdNm"),
                "addressDetail": r.get("vnaraExmnnDtadd"),
                "riskType": r.get("vnaraExmnnRgstrTpcdNm"),
            },
        })

    print(f"좌표 없어서 제외: {skipped_no_coord}건 / 북한산 경계 안: {len(features)}건")

    geojson = {"type": "FeatureCollection", "features": features}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
