"""Map View with Leaflet integration for Phase 2."""
import json
import os
from typing import Optional, List

from PySide6.QtCore import Qt, QObject, Slot, Signal, Property
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QPushButton,
    QComboBox, QLabel, QSlider,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel


class _MapBridge(QObject):
    """Bridge for Python <-> JavaScript communication via QWebChannel."""
    nav_data_received = Signal(str)
    map_clicked = Signal(float, float)
    polygon_drawn = Signal(str)
    page_ready = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._nav_json = "[]"

    @Property(str, notify=nav_data_received)
    def navData(self):
        return self._nav_json

    @navData.setter
    def navData(self, value):
        if value != self._nav_json:
            self._nav_json = value
            self.nav_data_received.emit(value)

    @Slot(float, float)
    def onMapClicked(self, lat, lon):
        self.map_clicked.emit(lat, lon)

    @Slot(str)
    def onPolygonDrawn(self, geojson):
        self.polygon_drawn.emit(geojson)

    @Slot()
    def onPageReady(self):
        self.page_ready.emit()


def _leaflet_html() -> str:
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>pyat Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>html,body,#map{margin:0;padding:0;width:100%;height:100%}#map{background:#1e1e1e}
.leaflet-control-zoom a{background:#2d2d2d!important;color:#d4d4d4!important}</style></head>
<body><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
var map=L.map('map',{zoomControl:true}).setView([0,0],2);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'(c) OSM'}).addTo(map);
var navLayer=L.layerGroup().addTo(map);
var highlightLayer=L.layerGroup().addTo(map);
var polygonLayer=L.layerGroup().addTo(map);
var bridge=null;
var drawing=false,drawPoints=[],drawLine=null;

new QWebChannel(qt.webChannelTransport,function(ch){
    bridge=ch.objects.bridge;
    map.on('click',function(e){
        if(drawing){drawPoints.push(e.latlng);
        if(drawLine)map.removeLayer(drawLine);
        if(drawPoints.length>1)drawLine=L.polyline(drawPoints,{color:'#ff6600',weight:2,dashArray:'5,5'}).addTo(map)}
        else bridge.onMapClicked(e.latlng.lat,e.latlng.lng)});
    map.on('dblclick',function(e){if(drawing&&drawPoints.length>=3)toggleDraw()});
    bridge.onPageReady();
});
function toggleDraw(){
    drawing=!drawing;
    if(!drawing&&drawPoints.length>=3){
        var gj={type:"Polygon",coordinates:[drawPoints.map(function(p){return[p.lng,p.lat]})]};
        bridge.onPolygonDrawn(JSON.stringify(gj));
        L.polygon(drawPoints,{color:'#ff6600',fillOpacity:.15}).addTo(polygonLayer)}
    if(!drawing){drawPoints=[];if(drawLine){map.removeLayer(drawLine);drawLine=null}}
    map.getContainer().style.cursor=drawing?'crosshair':''}
function toggleROIDraw(){toggleDraw()}
</script></body></html>"""


class MapView(QWidget):
    """Leaflet-based map view embedded in QWebEngineView."""
    map_clicked = Signal(float, float)
    polygon_drawn = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._bridge = _MapBridge(self)
        self._nav_tracks: dict = {}        # filepath -> coords (list of [lat,lon])
        self._visible_tracks: set = set()   # filepaths of visible tracks
        self._page_ready = False
        self._pending_js: List[str] = []
        self._setup_ui()
        self._bridge.page_ready.connect(self._on_page_ready)

    def _on_page_ready(self) -> None:
        self._page_ready = True
        for cmd in self._pending_js:
            self._web.page().runJavaScript(cmd)
        self._pending_js.clear()

    def _run_js(self, code: str) -> None:
        if self._page_ready:
            self._web.page().runJavaScript(code)
        else:
            self._pending_js.append(code)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        tb = QToolBar()
        self._roi_btn = QPushButton("ROI Polygon")
        self._roi_btn.setCheckable(True)
        self._roi_btn.toggled.connect(lambda c: self._run_js("toggleROIDraw()"))
        tb.addWidget(self._roi_btn)
        tb.addWidget(QPushButton("Fit All", clicked=lambda: self._run_js("try{map.fitBounds(map.getBounds())}catch(e){}")))
        layout.addWidget(tb)

        # Color Ramp + dB range toolbar
        cr_tb = QToolBar()
        cr_tb.addWidget(QLabel("Color:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Gray", "Jet"])
        self._cmap_combo.setCurrentText("Viridis")
        self._cmap_combo.currentTextChanged.connect(self._on_color_changed)
        cr_tb.addWidget(self._cmap_combo)

        cr_tb.addWidget(QLabel("dB Min:"))
        self._db_min = QSlider(Qt.Horizontal)
        self._db_min.setRange(-80, 0)
        self._db_min.setValue(-60)
        self._db_min.setToolTip("dB Minimum (clamp below)")
        self._db_min.valueChanged.connect(self._on_range_changed)
        cr_tb.addWidget(self._db_min)

        cr_tb.addWidget(QLabel("dB Max:"))
        self._db_max = QSlider(Qt.Horizontal)
        self._db_max.setRange(-80, 0)
        self._db_max.setValue(-10)
        self._db_max.setToolTip("dB Maximum (clamp above)")
        self._db_max.valueChanged.connect(self._on_range_changed)
        cr_tb.addWidget(self._db_max)

        self._db_label = QLabel("[-60, -10] dB")
        cr_tb.addWidget(self._db_label)
        layout.addWidget(cr_tb)

        self._web = QWebEngineView()
        self._web.setMinimumHeight(300)

        self._channel = QWebChannel()
        self._channel.registerObject("bridge", self._bridge)
        self._web.page().setWebChannel(self._channel)
        self._web.setHtml(_leaflet_html())

        self._bridge.map_clicked.connect(self.map_clicked)
        self._bridge.polygon_drawn.connect(self.polygon_drawn)

        layout.addWidget(self._web)

    def _run_js(self, code: str) -> None:
        self._web.page().runJavaScript(code)

    def add_file_track(self, filepath: str, coords: list) -> None:
        """Store nav track and draw it (visible by default)."""
        if not coords:
            return
        self._nav_tracks[filepath] = coords
        self._visible_tracks.add(filepath)
        self._redraw_all()

    def show_file_track(self, filepath: str, visible: bool) -> None:
        """Show/hide a file's nav track, then redraw."""
        if visible:
            self._visible_tracks.add(filepath)
        else:
            self._visible_tracks.discard(filepath)
        self._redraw_all()

    def _redraw_all(self) -> None:
        """Clear navLayer and redraw all visible tracks."""
        if not self._nav_tracks:
            return
        parts = ["navLayer.clearLayers()"]
        for fp, coords in self._nav_tracks.items():
            if fp not in self._visible_tracks:
                continue
            pts = ",".join(f"[{c[0]},{c[1]}]" for c in coords)
            parts.append(
                f"L.polyline([{pts}],{{color:'#4ec94e',weight:1,opacity:.5}}).addTo(navLayer)"
            )
        self._run_js(";".join(parts))

    def draw_nav_line(self, coords: list, color: str = "#ffcc00", weight: int = 3) -> None:
        """Highlight a single navigation line. Does NOT touch navLayer."""
        if not coords:
            return
        pts = ",".join(f"[{c[0]},{c[1]}]" for c in coords)
        self._run_js(
            f"highlightLayer.clearLayers();"
            f"L.polyline([{pts}],{{color:'{color}',weight:{weight},opacity:0.9}}).addTo(highlightLayer);"
        )

    def add_polygon_overlay(self, geojson_str: str, color: str = "#ff6600") -> None:
        self._run_js(f"try{{var gj=JSON.parse('{geojson_str}');var c=gj.coordinates[0].map(function(p){{return[p[1],p[0]]}});L.polygon(c,{{color:'{color}',fillOpacity:.15}}).addTo(polygonLayer)}}catch(e){{}}")

    def clear_overlays(self) -> None:
        self._run_js("navLayer.clearLayers();highlightLayer.clearLayers();polygonLayer.clearLayers()")

    def fly_to_bounds(self, lats: list, lons: list) -> None:
        """Fly (animated zoom) to the bounding box of given coordinates."""
        if not lats or not lons:
            return
        sw_lat = min(lats)
        sw_lon = min(lons)
        ne_lat = max(lats)
        ne_lon = max(lons)
        js = (
            f"map.flyToBounds([[{sw_lat},{sw_lon}],[{ne_lat},{ne_lon}]],"
            f"{{padding:[30,30],maxZoom:14,duration:1.5}})"
        )
        self._run_js(js)

    def draw_nav_line(self, coords: list, color: str = "#ffcc00", weight: int = 3) -> None:
        """Draw a single highlighted navigation line on the map.

        coords: list of [lat, lon] pairs.
        color: CSS color for the line (default yellow for highlights).
        weight: line thickness in pixels.
        """
        if not coords:
            return
        pts = ",".join(f"[{c[0]},{c[1]}]" for c in coords)
        js = (
            f"navLayer.clearLayers();"
            f"L.polyline([{pts}],{{color:'{color}',weight:{weight},opacity:0.9}}).addTo(navLayer);"
        )
        self._run_js(js)

    def _on_color_changed(self, text: str) -> None:
        """Placeholder for future GeoTIFF color ramp application."""
        pass

    def _on_range_changed(self) -> None:
        mn = self._db_min.value()
        mx = self._db_max.value()
        self._db_label.setText(f"[{mn}, {mx}] dB")
