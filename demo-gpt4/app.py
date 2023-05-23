import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collage import create_collage

app = Flask(__name__)

selected_images = {}

patch_size = 14 #16
patch_count = 16 #14
image_rows = 3
image_columns = 2
image_size = 224

selected_patches = []

distances = np.random.rand(image_rows*image_columns*patch_count*patch_count, image_rows*image_columns*patch_count*patch_count)

# Create a custom colormap with transparent 0s and semitransparent blue 1s
colors = [(0, 0, 1, 0.1), (0, 0, 1, 0.8)]  # RGBA: (transparent, semitransparent blue)
cmap_train = mcolors.LinearSegmentedColormap.from_list('selected_patches', colors, N=2)

colors = [(1, 0, 0, 0), (1, 0, 0, 1)]  # RGBA: (transparent, semitransparent red)
cmap_test = mcolors.LinearSegmentedColormap.from_list('heatmap', colors, N=256)



def cosine_distance_torch(x1, x2=None):
    x2 = x1 if x2 is None else x2
    return torch.mm(x1.to(float), x2.t().to(float))

def calculate_distances():
    print("Loading embeddings")

    # train_embeds = np.load('../embeddings/fair1m/f1m_dino_8shot_embeds_train.npy', allow_pickle=True) #dinov1
    # val_embeds = np.load('../embeddings/fair1m/f1m_dino_8shot_embeds_val.npy', allow_pickle=True) #dinov1

    train_embeds = np.load('../embeddings/fair1m/dinov2_250_8shot_embeds_train.npy', allow_pickle=True) #dinov2
    val_embeds = np.load('../embeddings/fair1m/dinov2_250_8shot_embeds_val.npy', allow_pickle=True) #dinov2
    
    train_selected_embedings = []
    for ind in selected_images['train']:
        train_selected_embedings.append(train_embeds[ind*patch_count*patch_count:(ind+1)*patch_count*patch_count])

    val_selected_embedings = []
    for ind in selected_images['val']:
        val_selected_embedings.append(val_embeds[ind*patch_count*patch_count:(ind+1)*patch_count*patch_count])
        
    print("Converting to torch tensors")
    train_selected_embedings = torch.tensor(train_selected_embedings)
    print(train_selected_embedings.shape)
    train_selected_embedings = train_selected_embedings.reshape((image_rows, image_columns, patch_count, patch_count, -1)).permute((0,2,1,3,4)).reshape((image_rows * patch_count * image_columns * patch_count, -1))
    
    val_selected_embedings = torch.tensor(val_selected_embedings)
    val_selected_embedings = val_selected_embedings.reshape((image_rows, image_columns, patch_count, patch_count, -1)).permute((0,2,1,3,4)).reshape((image_rows * patch_count * image_columns * patch_count, -1))
    
    print("Calculating the distances")
    distances[:,:] = cosine_distance_torch(train_selected_embedings, val_selected_embedings)
    
    print("Done")

@app.route('/')
def index():
    return render_template('index.html')

def update_overlay_train():
    data = np.zeros((image_rows * patch_count * image_columns * patch_count))
    data[selected_patches] = 1
    data = data.reshape((image_rows * patch_count, image_columns * patch_count))

    # Desired output image size in pixels
    width_px = image_columns * image_size
    height_px = image_rows * image_size

    # Calculate figure size in inches (width, height)
    dpi = 100  # You can adjust the dpi value for higher or lower resolution
    figsize = (width_px / dpi, height_px / dpi)

    # Create a figure with the specified size
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Remove axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Plot the binary matrix using imshow()
    ax.imshow(data, cmap=cmap_train, extent=[0, width_px, 0, height_px])

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the plot as a transparency-enabled PNG
    plt.savefig('static/img/overlay1.png', transparent=True, dpi=130, bbox_inches='tight', pad_inches=0)


def update_overlay_test():  
    data = distances[selected_patches].max(axis=0)
    data = data.reshape((image_rows * patch_count, image_columns * patch_count))
    print("test overlay shape", data.shape)
    print(data)
    
    # Desired output image size in pixels
    width_px = image_columns * image_size
    height_px = image_rows * image_size

    # Calculate figure size in inches (width, height)
    dpi = 100  # You can adjust the dpi value for higher or lower resolution
    figsize = (width_px / dpi, height_px / dpi)

    # Create a figure with the specified size
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Remove axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Plot the binary matrix using imshow()
    print(data.min(), data.mean(), data.max())
    vmin = min(1500, data.max()*.95)
    ax.imshow(data, cmap=cmap_test, extent=[0, width_px, 0, height_px], vmin=vmin)

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the plot as a transparency-enabled PNG
    plt.savefig('static/img/overlay2.png', transparent=True, dpi=130, bbox_inches='tight', pad_inches=0)

 
@app.route('/update_images', methods=['POST'])
def update_images():
    
    x = int(request.json['x'])
    y = int(request.json['y'])
    
    x_index = x // patch_size
    y_index = y // patch_size
    print(x, y, x_index, y_index)
    index = y_index * patch_count * image_columns + x_index
    
    if index in selected_patches:
        selected_patches.remove(index)
    else:
        selected_patches.append(index)
        
    print("Currently selected patches", selected_patches)
    update_overlay_train()
    update_overlay_test()
    
#     x_image_index = x_total_index // patch_count
#     y_image_index = y_total_index // patch_count
    
#     x_patch_index = x_total_index % patch_count
#     y_patch_index = y_total_index % patch_count
    
    

    return jsonify({'success': True})

if __name__ == '__main__':
    print("Creating the collage")
    t, v = create_collage('../images/f1m/250_8shot/train/train.npy', '../images/f1m/250_8shot/val/val.npy',
                   '../images/f1m/250_8shot/')
    print(f"Selected train images: {t}, test images: {v}")
    selected_images['train'] = t
    selected_images['val'] = v
    
    calculate_distances()
    app.run(debug=True, port=8050, host='0.0.0.0')
