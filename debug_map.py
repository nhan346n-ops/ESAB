import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineCore import QWebEnginePage
from gui.views.map_view import MapView
from PySide6.QtCore import QTimer

app = QApplication(sys.argv)

class LoggingPage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, msg, line, sourceID):
        print(f"JS LOG [{level}]: {msg} (at {sourceID}:{line})")

map_view = MapView()
page = LoggingPage(map_view._web)
page.setWebChannel(map_view._channel)
map_view._web.setPage(page)
map_view._web.setUrl(map_view._html_path) # reload

def quit_app():
    print("Done waiting.")
    app.quit()

QTimer.singleShot(3000, quit_app)
map_view.show()
sys.exit(app.exec())
