import splitfolders  # or import splitfolders
input_folder = 'dataset/Own_complete'
output = 'dataset/Own_complete_split' #where you want the split datasets saved. one will be created if it does not exist or none is set

# ratio of split are in order of train/val/test. You can change to whatever you want. For train/val sets only, you could do .75, .25 for example.
splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.7, .15, .15))
