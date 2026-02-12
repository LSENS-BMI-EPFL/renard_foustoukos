#!/usr/bin/env python3
"""
Compare two reactivation results files to check if they're different.
"""
import pickle
import numpy as np
import os

# Paths to the two files
results_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/reactivation'
file1 = os.path.join(results_dir, 'reactivation_results.pkl')
file2 = os.path.join(results_dir, 'reactivation_results_p99.pkl')

print("="*60)
print("COMPARING REACTIVATION RESULTS FILES")
print("="*60)

# Check if files exist
if not os.path.exists(file1):
    print(f"File 1 not found: {file1}")
    exit(1)
if not os.path.exists(file2):
    print(f"File 2 not found: {file2}")
    exit(1)

print(f"\nFile 1: {file1}")
print(f"File 2: {file2}")

# Load both files
with open(file1, 'rb') as f:
    data1 = pickle.load(f)

with open(file2, 'rb') as f:
    data2 = pickle.load(f)

print("\n" + "="*60)
print("STRUCTURE COMPARISON")
print("="*60)

# Check if they have the same keys
print(f"\nFile 1 keys: {list(data1.keys())}")
print(f"File 2 keys: {list(data2.keys())}")

if 'parameters' in data1:
    print(f"\nFile 1 parameters:")
    for k, v in data1['parameters'].items():
        print(f"  {k}: {v}")

if 'parameters' in data2:
    print(f"\nFile 2 parameters:")
    for k, v in data2['parameters'].items():
        print(f"  {k}: {v}")

# Compare R+ results
print("\n" + "="*60)
print("COMPARING R+ MICE RESULTS")
print("="*60)

r_plus_1 = data1.get('r_plus_results', {})
r_plus_2 = data2.get('r_plus_results', {})

print(f"\nFile 1: {len(r_plus_1)} mice")
print(f"File 2: {len(r_plus_2)} mice")

# Check a few mice in detail
common_mice = set(r_plus_1.keys()) & set(r_plus_2.keys())
print(f"\nCommon mice: {len(common_mice)}")

if common_mice:
    sample_mouse = list(common_mice)[0]
    print(f"\nDetailed comparison for mouse: {sample_mouse}")

    mouse_data_1 = r_plus_1[sample_mouse]
    mouse_data_2 = r_plus_2[sample_mouse]

    # Check if they have the same days
    days_1 = set(mouse_data_1.get('days', {}).keys())
    days_2 = set(mouse_data_2.get('days', {}).keys())
    print(f"  Days in file 1: {sorted(days_1)}")
    print(f"  Days in file 2: {sorted(days_2)}")

    # Compare specific day data
    if days_1 and days_2:
        sample_day = list(days_1 & days_2)[0]
        print(f"\n  Comparing day {sample_day}:")

        day_data_1 = mouse_data_1['days'][sample_day]
        day_data_2 = mouse_data_2['days'][sample_day]

        # Check number of events
        events_1 = day_data_1.get('events', [])
        events_2 = day_data_2.get('events', [])
        print(f"    Events in file 1: {len(events_1)}")
        print(f"    Events in file 2: {len(events_2)}")

        # Check threshold used
        if 'threshold_used' in day_data_1:
            print(f"    Threshold in file 1: {day_data_1['threshold_used']:.4f}")
        if 'threshold_used' in day_data_2:
            print(f"    Threshold in file 2: {day_data_2['threshold_used']:.4f}")

        # Check correlations (these should be the same - they're raw correlations)
        corr_1 = day_data_1.get('correlations', [])
        corr_2 = day_data_2.get('correlations', [])
        if len(corr_1) > 0 and len(corr_2) > 0:
            print(f"    Correlation arrays equal: {np.allclose(corr_1, corr_2)}")
            print(f"    Max correlation in file 1: {np.max(corr_1):.4f}")
            print(f"    Max correlation in file 2: {np.max(corr_2):.4f}")

        # Check session-level metrics
        if 'session_frequency' in day_data_1 and 'session_frequency' in day_data_2:
            print(f"    Session frequency file 1: {day_data_1['session_frequency']:.4f}")
            print(f"    Session frequency file 2: {day_data_2['session_frequency']:.4f}")

print("\n" + "="*60)
print("OVERALL ASSESSMENT")
print("="*60)

# Quick check if files are identical
identical = True
if len(r_plus_1) != len(r_plus_2):
    identical = False
    print("\n⚠️  FILES ARE DIFFERENT: Different number of R+ mice")
else:
    # Check if event counts differ for any mouse-day combination
    different_events = []
    for mouse in common_mice:
        if mouse in r_plus_1 and mouse in r_plus_2:
            days_1 = r_plus_1[mouse].get('days', {})
            days_2 = r_plus_2[mouse].get('days', {})
            for day in set(days_1.keys()) & set(days_2.keys()):
                events_1 = len(days_1[day].get('events', []))
                events_2 = len(days_2[day].get('events', []))
                if events_1 != events_2:
                    different_events.append((mouse, day, events_1, events_2))
                    identical = False

    if different_events:
        print(f"\n⚠️  FILES ARE DIFFERENT: Found {len(different_events)} mouse-day combinations with different event counts")
        print("\nSample differences:")
        for mouse, day, n1, n2 in different_events[:5]:
            print(f"  {mouse}, day {day}: {n1} events vs {n2} events")
    elif identical:
        print("\n⚠️  FILES APPEAR IDENTICAL!")
        print("This suggests the surrogate thresholds may not have been used during detection.")
        print("\nPossible reasons:")
        print("1. use_surrogate_thresholds was set to False")
        print("2. threshold_mode was set to 'mouse' instead of 'day'")
        print("3. The surrogate threshold CSV file was not found")
        print("4. The thresholds in the CSV happen to be very close to 0.45")

print("\n" + "="*60)
