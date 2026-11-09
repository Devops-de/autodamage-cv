"""
Entry point to launch the ClaimScope application.
Runs Uvicorn web server hosting FastAPI backend and static frontend.
"""
import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("  ClaimScope - Vehicle Damage & Collision Appraisal Engine")
    print("  Backend: OpenCV Signal Processing Pipeline")
    print("  Actuarial Standards: Mitchell / CCC ONE US Collision Guidelines")
    print("=" * 70)
    print("  Server launching at: http://localhost:8000")
    print("  API Docs available at: http://localhost:8000/docs")
    print("=" * 70)

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
