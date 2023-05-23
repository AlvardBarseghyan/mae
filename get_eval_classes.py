def get_eval_classes(dataset_name:str, file_path='./pre_process/labels.txt'):
    if dataset_name == 'cs':
        with open(file_path) as f:
            labels = {
                int(line[22:25].strip()):
                (line[:22].strip().replace("'", ""), 
                int(line[25:].strip()))
                for line in f.readlines()
            }

        eval_labels = [i for i in labels if 0 <= labels[i][1] < 255]
        eval_label_names = [labels[i][0] for i in labels if 0 <= labels[i][1] < 255]

        return eval_labels, eval_label_names
    
    elif dataset_name == 'f1':
        eval_labels = [0, 1, 2, 3, 4, 5]
        eval_label_names = ['background','airplane', 'ship', 'vehicle', 'court', 'roundabout']
        
        return eval_labels, eval_label_names
