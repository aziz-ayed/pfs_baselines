import pandas as pd
import numpy as np

# -------------------------------
# Load Data
# -------------------------------

# Follow-up data with recurrence/progression info
tcga_metadata = pd.read_csv(
    "/data/rbg/users/azizayed/chemistry/survival_baselines/data/follow_up.tsv",
    sep="\t"
)

# Clinical data with death info
tcga_metadata_2 = pd.read_csv(
    "/data/rbg/users/azizayed/chemistry/survival_baselines/data/clinical.tsv",
    sep="\t"
)

# -------------------------------
# Clean Columns
# -------------------------------

# Standardize NaNs in follow-up fields
for col in ['follow_ups.days_to_progression', 'follow_ups.days_to_recurrence', 'follow_ups.days_to_follow_up']:
    tcga_metadata[col] = pd.to_numeric(
        tcga_metadata[col].replace(" '-- ", np.nan), errors='coerce'
    )

# Clean death time column from clinical data
tcga_metadata_2['demographic.days_to_death'] = pd.to_numeric(
    tcga_metadata_2['demographic.days_to_death'].replace(" '-- ", np.nan), errors='coerce'
)

# -------------------------------
# Define Progression/Recurrence/Death Time per Patient
# -------------------------------

def get_pfs_event_time(group):
    prog_days = group['follow_ups.days_to_progression'].dropna()
    recur_days = group['follow_ups.days_to_recurrence'].dropna()
    yes_rows = group[group['follow_ups.progression_or_recurrence'] == 'Yes']
    yes_days = yes_rows['follow_ups.days_to_follow_up'].dropna()

    # Minimum of available event times
    combined = pd.concat([prog_days, recur_days, yes_days])
    return combined.min() if not combined.empty else np.nan

# Calculate time to progression/recurrence per patient
pfs_event_days = tcga_metadata.groupby('cases.submitter_id').apply(get_pfs_event_time)

# Calculate max follow-up per patient (for censoring)
max_follow_up_days = tcga_metadata.groupby('cases.submitter_id')['follow_ups.days_to_follow_up'].apply(
    lambda x: x.max()
)

# Get project_id per patient
project_ids = tcga_metadata.groupby('cases.submitter_id')['project.project_id'].first()

# -------------------------------
# Merge with Death Data
# -------------------------------

# Create death time series
death_days = tcga_metadata_2.groupby('cases.submitter_id')['demographic.days_to_death'].max()

# Align indexes (ensure all patient IDs are consistent)
all_patient_ids = sorted(set(pfs_event_days.index).union(set(death_days.index)))

# Combine time-to-event options: progression/recurrence OR death
combined_event_time = pd.Series(index=all_patient_ids, dtype='float')
for patient in all_patient_ids:
    times = []
    if patient in pfs_event_days and not pd.isna(pfs_event_days[patient]):
        times.append(pfs_event_days[patient])
    if patient in death_days and not pd.isna(death_days[patient]):
        times.append(death_days[patient])
    combined_event_time[patient] = min(times) if times else np.nan

# Get censoring times
combined_follow_up_time = pd.Series(index=all_patient_ids, dtype='float')
for patient in all_patient_ids:
    times = []
    if patient in max_follow_up_days and not pd.isna(max_follow_up_days[patient]):
        times.append(max_follow_up_days[patient])
    if patient in death_days and not pd.isna(death_days[patient]):
        times.append(death_days[patient])
    combined_follow_up_time[patient] = max(times) if times else np.nan

# Define event flag: 1 if event (progression/recurrence/death), 0 if censored
event_flags = combined_event_time.notna().astype(int)

# -------------------------------
# Final Dataset Construction
# -------------------------------

final_dataset = pd.DataFrame({
    'patient_id': combined_event_time.index,
    'project_id': project_ids.reindex(combined_event_time.index).values,
    'days_to_progression_recurrence': combined_event_time.values,
    'max_follow_up_days': combined_follow_up_time.values,
    'progression_recurrence_event': event_flags.values
})

zero_day_event_mask = (
    (final_dataset['progression_recurrence_event'] == 1) &
    (final_dataset['days_to_progression_recurrence'] == 0.0)
)

num_removed = zero_day_event_mask.sum()
print(f"\nRemoving {num_removed} 0-day events from final dataset.")

# Drop those rows
final_dataset = final_dataset[~zero_day_event_mask]

# -------------------------------
# Preview
# -------------------------------

print("Clean final dataset:")
print(final_dataset.head(10))
print(f"\nColumns: {list(final_dataset.columns)}")
print(f"Dataset shape: {final_dataset.shape}")
print(f"Patients with event (1): {final_dataset['progression_recurrence_event'].sum()}")
print(f"Patients without event (0): {(final_dataset['progression_recurrence_event'] == 0).sum()}")

final_dataset.to_csv("/data/rbg/users/azizayed/chemistry/survival_baselines/data/clinical_data.csv")