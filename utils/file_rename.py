import os

def rename_split(folder_path, x):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
    files.sort()

    first = files[:x]
    rest = files[x:]

    for i, filename in enumerate(first, start=1):
        old_path = os.path.join(folder_path, filename)
        new_filename = f"image_{i}.jpg"
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"{filename} -> {new_filename}")

    for i, filename in enumerate(rest, start=1):
        old_path = os.path.join(folder_path, filename)
        new_filename = f"image_{i}_light.jpg"
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"{filename} -> {new_filename}")

if __name__ == "__main__":
    rename_split("./images/april_tag/", x=13)