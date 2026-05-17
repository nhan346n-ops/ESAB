"""Map View with Leaflet integration for Phase 2."""
import json, os, base64, tempfile, uuid
from typing import Optional, List

from PySide6.QtCore import Qt, QObject, Slot, Signal, Property, QUrl, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QPushButton,
    QComboBox, QLabel,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel


class _MapBridge(QObject):
    """Bridge for Python <-> JavaScript communication via QWebChannel."""
    nav_data_received = Signal(str)
    map_clicked = Signal(float, float)
    polygon_drawn = Signal(str)
    page_ready = Signal()
    mouse_position = Signal(float, float)  # lat, lon

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

    @Slot(float, float)
    def onMouseMove(self, lat, lon):
        self.mouse_position.emit(lat, lon)

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
.leaflet-control-zoom a{background:#2d2d2d!important;color:#d4d4d4!important}
.leaflet-image-layer img{image-rendering:auto;-ms-interpolation-mode:bicubic}
#coords-overlay{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);z-index:1000;background:rgba(30,30,30,0.85);color:#d4d4d4;padding:4px 10px;font-family:sans-serif;font-size:12px;border-radius:4px;border:1px solid #444;pointer-events:none;min-width:420px;text-align:center;white-space:nowrap}
#legend-container{position:absolute;bottom:15px;right:10px;z-index:1000;background:transparent;color:#ffffff;text-shadow:0 0 3px #000000;padding:8px;font-family:sans-serif;font-size:11px;display:none;width:110px}
#legend-title{font-weight:bold;margin-bottom:6px;font-size:10px;text-align:center;white-space:nowrap}
#legend-bar{width:12px;height:100px;border:1px solid #666}
#legend-labels{height:100px;position:relative;width:80px;display:inline-block;vertical-align:top;margin-left:8px}
.legend-val{position:absolute;left:0;white-space:nowrap;font-family:monospace;font-size:10px}
#legend-val-max{top:0;transform:translateY(-50%)}
#legend-val-mid{top:50%;transform:translateY(-50%)}
#legend-val-min{bottom:0;transform:translateY(50%)}
</style></head>
<body><div id="map"></div>
<div id="coords-overlay">\u7ecf\u7eac\u5ea6: --, --</div>
<div id="legend-container">
    <div id="legend-title">\u56fe\u4e8b</div>
    <div style="display:flex;flex-direction:row;align-items:center;justify-content:center;height:100px">
        <div id="legend-bar"></div>
        <div id="legend-labels">
            <div id="legend-val-max" class="legend-val">--</div>
            <div id="legend-val-mid" class="legend-val">--</div>
            <div id="legend-val-min" class="legend-val">--</div>
        </div>
    </div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
var map=L.map('map',{zoomControl:true}).setView([30,110],5);
var tk='351963c8b638ff7517f17374145c6115';
var sub=['0','1','2','3','4','5','6','7'];
var wmts='SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=';
var tdt={};
tdt.vec={base:L.tileLayer('https://t{s}.tianditu.gov.cn/vec_w/wmts?LAYER=vec&'+wmts+tk,{subdomains:sub,maxZoom:18,crossOrigin:'anonymous',attribution:'\u5929\u5730\u56fe\u77e2\u91cf'})};
tdt.vec.annot=L.tileLayer('https://t{s}.tianditu.gov.cn/cva_w/wmts?LAYER=cva&'+wmts+tk,{subdomains:sub,maxZoom:18,crossOrigin:'anonymous',attribution:'\u5929\u5730\u56fe\u77e2\u91cf\u6ce8\u8bb0'});
tdt.img={base:L.tileLayer('https://t{s}.tianditu.gov.cn/img_w/wmts?LAYER=img&'+wmts+tk,{subdomains:sub,maxZoom:18,crossOrigin:'anonymous',attribution:'\u5929\u5730\u56fe\u5f71\u50cf'})};
tdt.img.annot=L.tileLayer('https://t{s}.tianditu.gov.cn/cia_w/wmts?LAYER=cia&'+wmts+tk,{subdomains:sub,maxZoom:18,crossOrigin:'anonymous',attribution:'\u5929\u5730\u56fe\u5f71\u50cf\u6ce8\u8bb0'});
tdt.ter={base:L.tileLayer('https://t{s}.tianditu.gov.cn/ter_w/wmts?LAYER=ter&'+wmts+tk,{subdomains:sub,maxZoom:18,crossOrigin:'anonymous',attribution:'\u5929\u5730\u56fe\u5730\u5f62'})};
tdt.ter.annot=L.tileLayer('https://t{s}.tianditu.gov.cn/cta_w/wmts?LAYER=cta&'+wmts+tk,{subdomains:sub,maxZoom:18,crossOrigin:'anonymous',attribution:'\u5929\u5730\u56fe\u5730\u5f62\u6ce8\u8bb0'});
var activeMap='vec';
tdt.vec.base.addTo(map);tdt.vec.annot.addTo(map);
function switchBaseMap(t){
    if(t===activeMap)return;
    tdt[activeMap].base.remove();tdt[activeMap].annot.remove();
    tdt[t].base.addTo(map);tdt[t].annot.addTo(map);
    activeMap=t;
}
var dtmLayer=L.layerGroup().addTo(map);
var navLayer=L.layerGroup().addTo(map);
var highlightLayer=L.layerGroup().addTo(map);
var polygonLayer=L.layerGroup().addTo(map);
var bridge=null;
new QWebChannel(qt.webChannelTransport,function(ch){bridge=ch.objects.bridge});
var drawing=false,drawPoints=[],drawLine=null;
var dtmOverlays={};
var dtmVisible={};
var activeDtmLayer='backscatter';

function setDtmOverlay(fp,type,imageUrl,bounds){
    try{
        var old=dtmOverlays[fp]&&dtmOverlays[fp][type];
        if(old){dtmLayer.removeLayer(old)}
        var o=L.imageOverlay(imageUrl,bounds,{opacity:.9,className:'dtm-overlay'});
        if(!dtmOverlays[fp])dtmOverlays[fp]={};
        dtmOverlays[fp][type]=o;
        if(dtmVisible[fp]!==false&&activeDtmLayer===type){dtmLayer.addLayer(o)}
    }catch(e){}
}
function showDtmFile(fp,vis){
    dtmVisible[fp]=vis;
    try{
        var o=dtmOverlays[fp];
        if(!o||!o[activeDtmLayer])return;
        if(vis){o[activeDtmLayer].addTo(dtmLayer)}else{o[activeDtmLayer].removeFrom(dtmLayer)}
    }catch(e){}
}
function showDtmLayer(type){
    try{
        var prev=activeDtmLayer;
        activeDtmLayer=type;
        for(var fp in dtmOverlays){
            var o=dtmOverlays[fp];
            if(!o)continue;
            if(prev&&o[prev]){dtmLayer.removeLayer(o[prev])}
            if(o[type]&&dtmVisible[fp]!==false){dtmLayer.addLayer(o[type])}
        }
    }catch(e){}
}
function updateCoordsAndValue(lon,lat,valText){
    var txt='\u7ecf\u5ea6: '+lon.toFixed(6)+'\u00b0  \u7eac\u5ea6: '+lat.toFixed(6)+'\u00b0';
    if(valText){
        txt+=' | '+valText;
    }
    document.getElementById('coords-overlay').innerText=txt;
}
function updateLegend(layerType,vmin,vmax){
    var container=document.getElementById('legend-container');
    var title=document.getElementById('legend-title');
    var bar=document.getElementById('legend-bar');
    var valMax=document.getElementById('legend-val-max');
    var valMid=document.getElementById('legend-val-mid');
    var valMin=document.getElementById('legend-val-min');
    if(layerType==='backscatter'){
        title.innerText='\u540e\u5411\u6563\u5c04 (dB)';
        bar.style.background='linear-gradient(to top,#000000,#ffffff)';
        valMax.innerText=vmax.toFixed(1)+' dB';
        valMid.innerText=((vmin+vmax)/2).toFixed(1)+' dB';
        valMin.innerText=vmin.toFixed(1)+' dB';
        container.style.display='block';
    }else if(layerType==='elevation'){
        title.innerText='\u6c34\u6df1 (m)';
        bar.style.background='linear-gradient(to top,#3333a9,#0076a3,#00a400,#eedd82,#8b5a2b,#ffffff)';
        valMax.innerText=vmax.toFixed(1)+' m';
        valMid.innerText=((vmin+vmax)/2).toFixed(1)+' m';
        valMin.innerText=vmin.toFixed(1)+' m';
        container.style.display='block';
    }else{
        container.style.display='none';
    }
}
function hideLegend(){
    document.getElementById('legend-container').style.display='none';
}
map.on('mousemove',function(e){
    var lat=e.latlng.lat,lon=e.latlng.lng;
    if(bridge)bridge.onMouseMove(lat,lon);
});
</script></body></html>"""


class MapView(QWidget):
    """Leaflet-based map view embedded in QWebEngineView."""
    map_clicked = Signal(float, float)
    polygon_drawn = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._bridge = _MapBridge(self)
        self._nav_tracks: dict = {}
        self._visible_tracks: set = set()
        self._page_ready = False
        self._pending_js: List[str] = []
        self._dtm_files: dict = {}  # filepath -> {backscatter/elevation: bounds}
        self._dtm_gamma: float = 1.0
        self._pending_gamma: float = 1.0
        self._dtm_shoulder: float = 0.0
        self._pending_shoulder: float = 0.0
        self._gamma_timer = QTimer(self)
        self._gamma_timer.setSingleShot(True)
        self._gamma_timer.timeout.connect(self._do_apply_gamma)
        self._active_dtm_layer: Optional[str] = "backscatter"
        self._dtm_visible: dict = {}
        self._dtm_cache: dict = {}
        self._setup_ui()
        self._bridge.page_ready.connect(self._on_page_ready)

    def _on_page_ready(self) -> None:
        self._page_ready = True
        for cmd in self._pending_js:
            self._web.page().runJavaScript(cmd)
        self._pending_js.clear()
        self._update_legend_display()

    def _run_js(self, code: str) -> None:
        if self._page_ready:
            self._web.page().runJavaScript(code)
        else:
            self._pending_js.append(code)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        tb = QToolBar()
        tb.addWidget(QLabel("\u5e95\u56fe:"))
        self._map_combo = QComboBox()
        self._map_combo.addItems(["\u77e2\u91cf\u5e95\u56fe", "\u5f71\u50cf\u5e95\u56fe", "\u5730\u5f62\u6655\u6682"])
        self._map_combo.setCurrentText("\u77e2\u91cf\u5e95\u56fe")
        self._map_combo.currentTextChanged.connect(self._on_map_type_changed)
        tb.addWidget(self._map_combo)
        tb.addSeparator()
        self._roi_btn = QPushButton("ROI \u591a\u8fb9\u5f62")
        self._roi_btn.setCheckable(True)
        self._roi_btn.toggled.connect(lambda c: self._run_js("toggleROIDraw()"))
        tb.addWidget(self._roi_btn)
        tb.addWidget(QPushButton("\u9002\u5408\u5168\u90e8", clicked=lambda: self._run_js("try{map.fitBounds(map.getBounds())}catch(e){}")))
        layout.addWidget(tb)

        self._web = QWebEngineView()
        self._web.setMinimumHeight(300)
        from PySide6.QtWebEngineCore import QWebEngineSettings
        ws = self._web.settings()
        ws.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)

        self._channel = QWebChannel()
        self._channel.registerObject("bridge", self._bridge)
        self._web.page().setWebChannel(self._channel)
        # Capture JS console messages for debugging
        self._web.page().javaScriptConsoleMessage = lambda level, msg, line, src: (
            print(f"[JS {'WARN' if level == 2 else 'ERROR' if level == 3 else 'LOG'}] {msg} (at {src}:{line})")
            if level >= 2 else None
        )
        # Save HTML to temp file and load via file:// URL to avoid about:blank restrictions
        self._html_path = os.path.join(tempfile.gettempdir(), "pyat_map.html")
        with open(self._html_path, "w", encoding="utf-8") as f:
            f.write(_leaflet_html())
        self._web.setUrl(QUrl.fromLocalFile(self._html_path))

        self._bridge.map_clicked.connect(self.map_clicked)
        self._bridge.polygon_drawn.connect(self.polygon_drawn)

        layout.addWidget(self._web)

        # Coordinates are now displayed inside the map as a floating overlay
        self._bridge.mouse_position.connect(self._on_mouse_move_query)

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
                f"L.polyline([{pts}],{{color:'#ff4444',weight:2,opacity:.7}}).addTo(navLayer)"
            )
        self._run_js(";".join(parts))

    def draw_nav_line(self, coords: list, color: str = "#ffcc00", weight: int = 3) -> None:
        """Highlight a single navigation line on the map — drawn on highlightLayer, does NOT touch navLayer.

        Args:
            coords: list of [lat, lon] pairs.
            color: CSS color for the line (default yellow for highlights).
            weight: line thickness in pixels.
        """
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
        self._run_js("if(typeof navLayer !== 'undefined') { navLayer.clearLayers();highlightLayer.clearLayers();polygonLayer.clearLayers(); }")

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



    def _on_map_type_changed(self, text: str) -> None:
        m = {"\u77e2\u91cf\u5e95\u56fe": "vec", "\u5f71\u50cf\u5e95\u56fe": "img", "\u5730\u5f62\u6655\u6682": "ter"}
        t = m.get(text)
        if t:
            self._run_js(f"switchBaseMap('{t}')")

    def _update_legend_display(self) -> None:
        if not hasattr(self, '_active_dtm_layer') or not self._active_dtm_layer:
            self._run_js("hideLegend();")
            return
            
        layer_type = self._active_dtm_layer
        
        vmins = []
        vmaxs = []
        for fp, layers_cache in getattr(self, '_dtm_cache', {}).items():
            if self._dtm_visible.get(fp, True) and layer_type in layers_cache:
                vmins.append(layers_cache[layer_type]['vmin'])
                vmaxs.append(layers_cache[layer_type]['vmax'])
                
        if vmins and vmaxs:
            overall_vmin = min(vmins)
            overall_vmax = max(vmaxs)
            import json
            self._run_js(f"updateLegend({json.dumps(layer_type)}, {overall_vmin}, {overall_vmax});")
        else:
            self._run_js("hideLegend();")

    def _on_mouse_move_query(self, lat: float, lon: float) -> None:
        if not hasattr(self, '_active_dtm_layer') or not self._active_dtm_layer:
            self._run_js(f"updateCoordsAndValue({lon}, {lat}, null);")
            return
            
        layer_type = self._active_dtm_layer
        val_text = None
        for fp, layers_cache in getattr(self, '_dtm_cache', {}).items():
            if not self._dtm_visible.get(fp, True) or layer_type not in layers_cache:
                continue
                
            cache = layers_cache[layer_type]
            grid_lon = cache['lon']
            grid_lat = cache['lat']
            grid_data = cache['data']
            transformer = cache['transformer']
            
            try:
                target_lon, target_lat = lon, lat
                if transformer:
                    target_lon, target_lat = transformer.transform(lon, lat)
                
                min_x, max_x = float(grid_lon.min()), float(grid_lon.max())
                min_y, max_y = float(grid_lat.min()), float(grid_lat.max())
                
                if not (min_x <= target_lon <= max_x and min_y <= target_lat <= max_y):
                    continue
                    
                import numpy as np
                idx_x = int(np.argmin(np.abs(grid_lon - target_lon)))
                idx_y = int(np.argmin(np.abs(grid_lat - target_lat)))
                
                val = grid_data[idx_y, idx_x]
                if np.isfinite(val):
                    if layer_type == "backscatter":
                        val_text = f"\u540e\u5411\u6563\u5c04: {val:.2f} dB"
                    elif layer_type == "elevation":
                        val_text = f"\u6c34\u6df1: {val:.2f} m"
                    break
            except Exception:
                pass
                
        import json
        self._run_js(f"updateCoordsAndValue({lon}, {lat}, {json.dumps(val_text)});")

    def set_dtm_overlay(self, filepath: str, layer_type: str, image_url: str, bounds: tuple) -> None:
        """Add a DTM image overlay via Blob URL."""
        if filepath not in self._dtm_files:
            self._dtm_files[filepath] = {}
        self._dtm_files[filepath][layer_type] = bounds

        # Track visibility
        if filepath not in self._dtm_visible:
            self._dtm_visible[filepath] = True

        # Cache data
        if filepath not in self._dtm_cache:
            self._dtm_cache[filepath] = {}
        try:
            from .dtm_renderer import _read_dtm_layer
            lon, lat, data, _, vmin, vmax = _read_dtm_layer(filepath, layer_type)
            
            # Setup coordinate transformer
            from netCDF4 import Dataset
            transformer = None
            with Dataset(filepath, "r") as ds:
                has_proj = "x" in ds.variables and "y" in ds.variables
                if has_proj:
                    try:
                        crs_wkt = getattr(ds.variables.get("crs"), "crs_wkt", None)
                        if crs_wkt:
                            from pyproj import CRS, Transformer
                            src_crs = CRS.from_wkt(crs_wkt)
                            tgt_crs = CRS.from_epsg(4326)
                            transformer = Transformer.from_crs(tgt_crs, src_crs, always_xy=True)
                    except Exception:
                        pass
            
            self._dtm_cache[filepath][layer_type] = {
                'lon': lon,
                'lat': lat,
                'data': data,
                'vmin': vmin,
                'vmax': vmax,
                'transformer': transformer
            }
            self._update_legend_display()
        except Exception as e:
            print(f"Error caching DTM layer {layer_type}: {e}")
        import json
        bounds_str = json.dumps([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        fp = json.dumps(filepath)
        # Extract base64 from data URL, create Blob in JS
        if image_url.startswith("data:") and "base64," in image_url:
            b64data = image_url.split("base64,", 1)[1]
            esc = json.dumps(b64data)
            js = (
                "try{"
                f"var raw=atob({esc});var arr=new Uint8Array(raw.length);"
                "for(var i=0;i<raw.length;i++)arr[i]=raw.charCodeAt(i);"
                "var blob=new Blob([arr],{type:'image/png'});"
                f"var url=URL.createObjectURL(blob);"
                f"setDtmOverlay({fp},{json.dumps(layer_type)},url,{bounds_str})"
                "}catch(e){}"
            )
            self._run_js(js)

    def show_dtm_file(self, filepath: str, visible: bool) -> None:
        """Show or hide a specific DTM file's overlay on the map."""
        self._dtm_visible[filepath] = visible
        self._run_js(f"showDtmFile({json.dumps(filepath)},{str(visible).lower()})")
        self._update_legend_display()

    def show_dtm_layer(self, layer_type: Optional[str]) -> None:
        """Switch active DTM layer type: 'elevation', 'backscatter', or None."""
        self._active_dtm_layer = layer_type
        import json
        js = f"showDtmLayer({json.dumps(layer_type)});"
        self._run_js(js)
        self._update_legend_display()

    def cleanup(self) -> None:
        """Remove temp files."""
        if hasattr(self, '_html_path') and os.path.exists(self._html_path):
            try:
                os.remove(self._html_path)
            except Exception:
                pass

    def set_dtm_gamma(self, gamma: float) -> None:
        """Request gamma update — debounced: only applies after 300ms of no changes."""
        self._pending_gamma = gamma
        self._gamma_timer.start(300)

    def set_dtm_shoulder(self, shoulder: float) -> None:
        """Request shoulder compression update — shares the gamma debounce timer."""
        self._pending_shoulder = shoulder
        self._gamma_timer.start(300)

    def _do_apply_gamma(self) -> None:
        """Actually re-render all DTM layers with pending gamma and shoulder values."""
        gamma = self._pending_gamma
        shoulder = self._pending_shoulder
        if gamma == self._dtm_gamma and shoulder == self._dtm_shoulder:
            return
        self._dtm_gamma = gamma
        self._dtm_shoulder = shoulder
        from .dtm_renderer import dtm_layer_to_file
        for fp, layers in self._dtm_files.items():
            for layer_type, bounds in layers.items():
                png_path, data_url, _ = dtm_layer_to_file(
                    fp, layer_type,
                    cmap="gray" if layer_type == "backscatter" else "terrain",
                    gamma=gamma,
                    shoulder=shoulder,
                    hillshade=(layer_type == "elevation")
                )
                self.set_dtm_overlay(fp, layer_type, data_url, bounds)


