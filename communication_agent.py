import pandas as pd
import numpy as np
import json

def analyze_predictions_expert(predictions, beat_locations, fs):
    """
    Calculates expert-level statistics, including HRV and complex patterns.
    """
    if predictions is None or predictions.size == 0 or len(beat_locations) < 2:
        return None

    # Basic Stats
    total_beats = len(predictions)
    beat_counts = pd.Series(predictions).value_counts()
    beat_percentages = (beat_counts / total_beats) * 100

    # Rhythm & HRV Analysis
    rr_intervals = np.diff(beat_locations) # Time between beats in samples
    rr_intervals_ms = (rr_intervals / fs) * 1000 # Convert to milliseconds
    avg_bpm = 60000 / np.mean(rr_intervals_ms)
    sdnn = np.std(rr_intervals_ms) # Standard deviation of intervals (a key HRV metric)

    # Complex Pattern Analysis
    pred_str = "".join(predictions)
    pvc_couplets = pred_str.count('VV')
    # Find runs of 3 or more 'V's (Ventricular Tachycardia)
    v_tach_runs = [run for run in pred_str.replace('N',' ').split() if 'VVV' in run]
    # Check for Bigeminy (N,V,N,V pattern)
    bigeminy_count = pred_str.count('NV')

    analysis = {
        "total_beats": total_beats,
        "beat_counts": beat_counts,
        "beat_percentages": beat_percentages,
        "avg_bpm": avg_bpm,
        "sdnn": sdnn,
        "pvc_couplets": pvc_couplets,
        "v_tach_runs": len(v_tach_runs),
        "bigeminy_suspicion": bigeminy_count > (total_beats / 4), # Heuristic
    }
    return analysis

def generate_doctor_report_expert(analysis):
    """Generates a structured dictionary for the clinical report."""
    if not analysis: return None

    pvc_burden = analysis['beat_percentages'].get('V', 0)
    key_findings = []

    if pvc_burden > 15:
        key_findings.append(f"High PVC burden detected ({pvc_burden:.1f}%). Further evaluation may be warranted.")
    elif pvc_burden > 1:
        key_findings.append(f"PVC burden noted at {pvc_burden:.1f}%.")

    if analysis['pvc_couplets'] > 0:
        key_findings.append(f"Ventricular Couplets Detected: {analysis['pvc_couplets']}")
    if analysis['v_tach_runs'] > 0:
        key_findings.append(f"!! URGENT: Non-Sustained V-Tach Runs Detected: {analysis['v_tach_runs']} !!")
    if analysis['bigeminy_suspicion']:
        key_findings.append("Ventricular Bigeminy suspected.")

    if 'L' in analysis['beat_counts']: key_findings.append("Left Bundle Branch Block morphology present.")
    if 'R' in analysis['beat_counts']: key_findings.append("Right Bundle Branch Block morphology present.")
    if 'A' in analysis['beat_counts']: key_findings.append("Atrial Premature Beats (APBs) detected.")

    if not key_findings:
        key_findings.append("Rhythm appears to be consistently normal.")

    # Convert NumPy types to standard Python types for JSON compatibility
    return {
        "total_beats": int(analysis['total_beats']),
        "avg_bpm": f"{analysis['avg_bpm']:.1f}",
        "sdnn": f"{analysis['sdnn']:.2f} ms",
        "tachycardia": bool(analysis['avg_bpm'] > 100),
        "bradycardia": bool(analysis['avg_bpm'] < 60),
        "beat_distribution": [
            {"type": beat_type, "count": int(count), "percentage": f"{analysis['beat_percentages'][beat_type]:.1f}%"}
            for beat_type, count in analysis['beat_counts'].items()
        ],
        "key_findings": key_findings
    }

def generate_patient_report_expert_clean(analysis):
    """Generates a structured dictionary for the patient report."""
    if not analysis: return None

    avg_bpm = analysis['avg_bpm']
    pvc_burden = analysis['beat_percentages'].get('V', 0)
    other_abnormalities = len(analysis['beat_counts']) > (1 + ('V' in analysis['beat_counts']))

    hr_text = f"Your average heart rate was {avg_bpm:.0f} beats per minute (BPM), which is within the typical resting range of 60-100 BPM."
    if avg_bpm > 100:
        hr_text = f"Your average heart rate was {avg_bpm:.0f} beats per minute (BPM), which is considered faster than the typical resting range (tachycardia)."
    elif avg_bpm < 60:
        hr_text = f"Your average heart rate was {avg_bpm:.0f} beats per minute (BPM), which is considered slower than the typical resting range (bradycardia)."

    rhythm_text = ""
    if analysis['v_tach_runs'] > 0:
        rhythm_text = "The analysis detected a short run of very fast, irregular heartbeats. This is a significant finding that requires prompt medical review. Please ensure your doctor sees this report."
    elif analysis['bigeminy_suspicion']:
        rhythm_text = "A specific pattern called 'bigeminy' was detected, where every other heartbeat was an extra beat. It's important to discuss this pattern with your doctor."
    elif analysis['pvc_couplets'] > 0:
        rhythm_text = "The analysis found some extra beats (PVCs) occurring in pairs. While occasional extra beats are common, paired ones are something your doctor will want to review."
    elif pvc_burden > 1:
        rhythm_text = "A few extra heartbeats (PVCs) were noted. These are very common and often harmless, but it's always good to keep your doctor informed."
    elif len(analysis['beat_counts']) > 1:
        rhythm_text = "A few minor variations in the shape of your heartbeats were noted. This is often normal."
    else:
        rhythm_text = "Your heartbeat showed a consistent shape and rhythm throughout this recording."

    return {
        "sections": [
            {"heading": "1. Average Heart Rate", "text": hr_text},
            {"heading": "2. Heart Rate Variability (HRV)", "text": "We also measured the variation in time between each heartbeat. Healthy hearts naturally have some variation. Your doctor can interpret this measurement in the context of your overall health."},
            {"heading": "3. Heartbeat Rhythm and Patterns", "text": rhythm_text}
        ],
        "disclaimer": "Disclaimer: This is an automated analysis, not a diagnosis. Please share these detailed findings with your healthcare provider."
    }