import pandas as pd

csv_path = "data_recommendme.csv"  # change if needed

REAL_EMAIL_ID = "U0002"
REAL_EMAIL = "wimharrisryden1@gmail.com"

def make_dummy_requester_email(uid: str) -> str:
    uid = str(uid).strip().lower()
    if not uid:
        uid = "unknown"
    return f"req_{uid}@demo.edu"

df = pd.read_csv(csv_path)

# Normalize column names
df.columns = [str(c).strip().lower() for c in df.columns]

# Ensure requester_id exists
if "requester_id" not in df.columns:
    raise ValueError("CSV must contain a requester_id column")

# Add requester_email column if missing
if "requester_email" not in df.columns:
    insert_at = df.columns.get_loc("requester_id") + 1
    df.insert(insert_at, "requester_email", "")

# Clean requester_id and requester_email
df["requester_id"] = df["requester_id"].fillna("").astype(str).str.strip().str.upper()
df["requester_email"] = df["requester_email"].fillna("").astype(str).str.strip()

# Fill requester_email row by row
def assign_requester_email(row):
    uid = row["requester_id"]
    existing_email = row["requester_email"].strip()

    # Always preserve/set your real email for U0002
    if uid == REAL_EMAIL_ID:
        return REAL_EMAIL

    # If some other row somehow already has your real email, replace it with dummy
    if existing_email.lower() == REAL_EMAIL.lower():
        return make_dummy_requester_email(uid)

    # For all other users, force demo email
    return make_dummy_requester_email(uid)

df["requester_email"] = df.apply(assign_requester_email, axis=1)

# Save back to the same file
df.to_csv(csv_path, index=False)

print("CSV updated successfully.\n")

print("Requester email samples:")
print(df[["requester_id", "requester_email"]].drop_duplicates().head(20))

print("\nRows for U0002:")
print(df.loc[df["requester_id"] == REAL_EMAIL_ID, ["requester_id", "requester_email"]].drop_duplicates())