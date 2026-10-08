from weather_alpha.config import load_config


def test_default_four_stations() -> None:
    cfg = load_config(project_root=".")
    assert set(cfg.selected_station_ids) == {"EHAM", "KLGA", "KLAX", "EGLC"}
    assert cfg.stations["amsterdam"].sources.knmi is True
    assert all(station.sources.nws_tgftp for station in cfg.stations.values())
