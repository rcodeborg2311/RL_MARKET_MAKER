#!/usr/bin/env python3
"""Launch live Plotly Dash dashboard."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.app import app

if __name__ == "__main__":
    print("Dashboard running at http://localhost:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
