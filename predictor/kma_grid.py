"""
위경도 <-> 기상청 단기예보 격자좌표(nx, ny) 변환.
기상청이 공개한 LCC(Lambert Conformal Conic) 격자 변환 공식을 그대로 구현한 것.
"""
import math

RE = 6371.00877  # 지구 반경(km)
GRID = 5.0  # 격자 간격(km)
SLAT1 = 30.0  # 투영 위도1(degree)
SLAT2 = 60.0  # 투영 위도2(degree)
OLON = 126.0  # 기준점 경도(degree)
OLAT = 38.0  # 기준점 위도(degree)
XO = 43  # 기준점 X좌표(GRID 단위)
YO = 136  # 기준점 Y좌표(GRID 단위)

DEGRAD = math.pi / 180.0


def latlon_to_grid(lat: float, lon: float) -> tuple[int, int]:
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD

    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = math.pow(sf, sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / math.pow(ro, sn)

    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / math.pow(ra, sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn

    x = ra * math.sin(theta) + XO + 0.5
    y = ro - ra * math.cos(theta) + YO + 0.5
    return int(x), int(y)


def grids_covering_geojson(geojson_path: str, step_deg: float = 0.005) -> list[tuple[int, int]]:
    """GeoJSON(Polygon/MultiPolygon) 폴리곤 "내부"를 촘촘히 스캔해서, 그 영역이 실제로
    걸치는 기상청 5km 격자(nx,ny)만 골라 정렬된 리스트로 돌려준다. 격자 자체가 5km
    간격이라 step_deg(약 500m)면 격자 경계를 놓치지 않고 충분히 촘촘함.

    예전엔 폴리곤의 바운딩 박스(사각형) 전체를 스캔했는데, 북한산처럼 실제 모양이
    반듯한 사각형이 아닌 공원은 바운딩 박스 구석에 걸릴 뿐 실제 경계 밖인 격자까지
    딸려와서(12개 중 4개), 필요 없는 API 호출이 그만큼 늘어나는 문제가 있었다.
    shapely로 "실제 폴리곤 안에 들어오는 점인지"까지 확인해서 걸러낸다.
    """
    import json
    from shapely.geometry import shape, Point
    from shapely.ops import unary_union

    with open(geojson_path, encoding="utf-8") as f:
        geo = json.load(f)

    geoms = [shape(feature.get("geometry", feature)) for feature in geo.get("features", [geo])]
    if not geoms:
        return []
    park = unary_union(geoms)
    minx, miny, maxx, maxy = park.bounds

    grids = set()
    lon = minx
    while lon <= maxx:
        lat = miny
        while lat <= maxy:
            if park.contains(Point(lon, lat)):
                grids.add(latlon_to_grid(lat, lon))
            lat += step_deg
        lon += step_deg
    return sorted(grids)
