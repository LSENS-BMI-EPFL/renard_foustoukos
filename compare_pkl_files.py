import pickle
import numpy as np

# Load both PKL files
with open('/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/reactivation/reactivation_results_p95.pkl', 'rb') as f:
    p95_data = pickle.load(f)

with open('/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/reactivation/reactivation_results_p999.pkl', 'rb') as f:
    p999_data = pickle.load(f)

print("="*60)
print("COMPARING REACTIVATION RESULTS FILES")
print("="*60)

# Compare R+ results
print("\nR+ mice comparison:")
print(f"p95: {len(p95_data['r_plus_results'])} mice")
print(f"p999: {len(p999_data['r_plus_results'])} mice")

# Check if number of events detected is different
for mouse in p95_data['r_plus_results'].keys():
    if mouse in p999_data['r_plus_results']:
        p95_events = p95_data['r_plus_results'][mouse]
        p999_events = p999_data['r_plus_results'][mouse]

        # Count total events across all days
        p95_total = sum(len(p95_events.get(day, {}).get('events', [])) for day in [-2, -1, 0, 1, 2])
        p999_total = sum(len(p999_events.get(day, {}).get('events', [])) for day in [-2, -1, 0, 1, 2])

        if p95_total != p999_total:
            print(f"  {mouse}: p95={p95_total} events, p999={p999_total} events")

# Compare R- results
print("\nR- mice comparison:")
print(f"p95: {len(p95_data['r_minus_results'])} mice")
print(f"p999: {len(p999_data['r_minus_results'])} mice")

for mouse in p95_data['r_minus_results'].keys():
    if mouse in p999_data['r_minus_results']:
        p95_events = p95_data['r_minus_results'][mouse]
        p999_events = p999_data['r_minus_results'][mouse]

        # Count total events across all days
        p95_total = sum(len(p95_events.get(day, {}).get('events', [])) for day in [-2, -1, 0, 1, 2])
        p999_total = sum(len(p999_events.get(day, {}).get('events', [])) for day in [-2, -1, 0, 1, 2])

        if p95_total != p999_total:
            print(f"  {mouse}: p95={p95_total} events, p999={p999_total} events")

# Check overall event counts
print("\n" + "="*60)
print("OVERALL EVENT COUNTS")
print("="*60)

p95_total_all = 0
p999_total_all = 0

for mouse_dict in [p95_data['r_plus_results'], p95_data['r_minus_results']]:
    for mouse in mouse_dict.keys():
        p95_total_all += sum(len(mouse_dict[mouse].get(day, {}).get('events', [])) for day in [-2, -1, 0, 1, 2])

for mouse_dict in [p999_data['r_plus_results'], p999_data['r_minus_results']]:
    for mouse in mouse_dict.keys():
        p999_total_all += sum(len(mouse_dict[mouse].get(day, {}).get('events', [])) for day in [-2, -1, 0, 1, 2])

print(f"Total events p95: {p95_total_all}")
print(f"Total events p999: {p999_total_all}")
print(f"Difference: {p95_total_all - p999_total_all}")

if p95_total_all == p999_total_all:
    print("\n⚠️  WARNING: Event counts are IDENTICAL!")
    print("This suggests the same thresholds were used for both analyses.")
