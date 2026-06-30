import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtCore import QTimer

app = QApplication(sys.argv)

class LoggingPage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, msg, line, sourceID):
        print(f"JS LOG [{level}]: {msg} (at {sourceID}:{line})")

page = LoggingPage()
page.setHtml("""
<html>
<body>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
    console.log("QWebChannel type: " + typeof QWebChannel);
</script>
</body>
</html>
""")

def quit_app():
    print("Done waiting.")
    app.quit()

QTimer.singleShot(2000, quit_app)
sys.exit(app.exec())
