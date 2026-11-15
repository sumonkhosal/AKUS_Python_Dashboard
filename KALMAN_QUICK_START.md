# Kalman tab — 5 minute quick start

1. Install Python 3.10+ and extract the ZIP.
2. Double-click `start_windows.bat`, or run:
   `py -m pip install -r requirements.txt` then `py -m streamlit run app.py`.
3. The original dashboard has a **Kalman research** tab at the far right.
4. **To work with your existing virtual carrier:** use the Experiment tab,
   select **Home both axes (simulation)** and **Run sequence to completion**,
   then use Manual motion to move to 100 mm and finish the move.
   Select **Current carrier simulation** in the Kalman tab, select `reference`
   or `filterpy`, and click **Analyze Q / R / S / K**.
5. **To work with your actual noise data:** select **Included real AKUS recording**.
   It includes 2–10 s stationary and 230–240 s motion excerpts of your
   November 5, 2026 file, not synthetic traces. Click Analyze.
6. **For the full 296-second Scope file:** select **Upload TwinCAT Scope CSV**, pick
   `Test_KalmanFilter.csv`, set the analysis start time (e.g. 230 s) and
   duration (e.g. 10 s). Upload a stationary calibration export if available.
7. Inspect R/Q, the time-dependent K, S, NIS, motion spectrum and coherence;
   export the time series and report for reproducibility.

Important warnings: The real-data Q fit may reach its optimization bound due to
model mismatch; the encoder signal is not independent ground truth, and a
stationary variance is not a complete dynamic measurement model. The dashboard
will **never send commands to your Beckhoff hardware**, and its virtual E-STOP
is not a safety function. More validation is necessary before using estimates
for closed-loop control.
