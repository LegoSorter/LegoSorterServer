import os

dir = r"F:\slawek\enh\final\test"
classes = None
labels = os.listdir(dir)

# labels2 = os.listdir(r"F:\slawek\final_dataset_prepared_447_classes\final_dataset\train")
#
# exit()
filename = "dataframe"
if classes:
    filename+= f"_{classes}"
with open(os.path.join(dir, f"{filename}.csv"), "w") as dst:
    dst.write("label,image_path, \n")
    if classes:
        labels = labels[0:classes]
    print(len(labels))
    for label in labels:
        if "." in label:
            continue
        files = os.listdir(os.path.join(dir, label))
        for file in files:
            f = os.path.abspath(os.path.join(dir, label, file))
            dst.write(F"{label},{f}, \n")


