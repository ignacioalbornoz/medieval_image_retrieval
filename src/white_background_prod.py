import os
import numpy as np
from tqdm import tqdm
from skimage import io, morphology, measure
from skimage.filters import threshold_otsu
from sklearn.decomposition import PCA
from dinov2_feature_extraction import min_max_scale, get_dense_descriptor, load_model
import torch
from PIL import Image
from skimage.transform import resize
from decimal import Decimal, getcontext



def pca_colorize_2(features, output_shape, pca):
    # Aplicar PCA como antes
    inverted = False
    remove = True
    rgb = pca.transform(features)
    rgb = min_max_scale(rgb)
    rgb = rgb.reshape(output_shape + (3,))

    # Set the precision you need
    getcontext().prec = 10

    mean_color = np.mean(rgb, axis=(0, 1))
    print(mean_color)
    dominant_color = np.argmax(mean_color)
    print(dominant_color)
    # Convert numpy.float32 to native Python float, then to Decimal
    mean_color_0 = Decimal(float(mean_color[0]))
    mean_color_2 = Decimal(float(mean_color[2]))

    # Calculate the difference using Decimal
    diff =  mean_color_2-mean_color_0

    # Calculate the difference using Decimal
    diff_2 =  mean_color_0 - mean_color_2

    # Aplicar umbral de Otsu al color dominante
    thresh = threshold_otsu(rgb[:, :, 0])
    rgb_mask = (rgb[:, :, 0] > thresh)*1

    # Invertir la máscara si el color dominante no es el primero
    if dominant_color == 0:
        print("invertir")
        inverted = True
        rgb_mask = 1 - rgb_mask

    if dominant_color == 2 :
        remove = False

    if dominant_color == 0 :
        remove = False
    rgb[:, :, 0] *= rgb_mask
    rgb[:, :, 1] *= rgb_mask
    rgb[:, :, 2] *= rgb_mask
    rgb = min_max_scale(rgb)
    print("remove", remove)
    return rgb, inverted, remove



def foreground_mask_2(attention_rgb, remove, use_bbox=True):
    # Crear la máscara básica
    attention_mask = attention_rgb.mean(axis=-1) > 0
    attention_mask = morphology.binary_dilation(attention_mask)
    if not remove:
        print("no remove")
        # Crear una matriz de unos con las mismas dimensiones que attention_mask
        return np.ones_like(attention_mask)

    if use_bbox:
        attention_labeled = measure.label(attention_mask)
        regions = measure.regionprops(attention_labeled)
        '''
        if inverted:
            # Crear una máscara invertida donde se mantenga todo excepto las bounding boxes
            inverted_mask = np.ones_like(attention_mask, dtype=bool)
            for props in regions:
                ymin, xmin, ymax, xmax = props.bbox
                inverted_mask[ymin:ymax, xmin:xmax] = False
            return inverted_mask
         else:
        '''
        
        # Proceso normal: expandir la máscara dentro de los bounding boxes
        for props in regions:
            ymin, xmin, ymax, xmax = props.bbox
            attention_mask[ymin:ymax, xmin:xmax] = True

    return attention_mask




def convert_image(image):
    with torch.no_grad():
        if image.dtype == bool:
            # Convertir directamente las máscaras booleanas
            result = (image * 255).astype('uint8')
        else:
            # Escalar la imagen al rango [0, 255] y convertirla a uint8
            image = (image - image.min()) / (image.max() - image.min()) * 255
            result = image.astype('uint8')
    
    # Liberar la memoria de la GPU
    torch.cuda.empty_cache()
    
    return result




def resize_image_pil(image, scale_factor):
    pil_image = Image.fromarray(np.uint8(image))
    new_dimensions = (int(pil_image.width * scale_factor), int(pil_image.height * scale_factor))
    resized_image = pil_image.resize(new_dimensions)
    torch.cuda.empty_cache()
    return np.array(resized_image)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    #dataset_path = r'D:\datasets\medieval_images\DocExplore_images'   
    dataset_path = r'/home/ignacio/2024-1/tesis/verano/medieval_image_retrieval/DocExplore_images'   
    save_path = r'/home/ignacio/2024-1/tesis/verano/medieval_image_retrieval/result/pca_mask'
    #save_path_mask = r'/home/ignacio/2024-1/tesis/medieval_image_retrieval/foreground_mask_2'

    dataset_path_white_img_1 = r'/home/ignacio/2024-1/tesis/verano/medieval_image_retrieval/DocExplore_images/page27.jpg'   
    #dataset_path_white_img_2 = r'/home/ignacio/2024-1/tesis/verano/medieval_image_retrieval/DocExplore_images/page1328.jpg'   
    img = io.imread(dataset_path_white_img_1)


    dinov2_sizes = {"small": 384,
                    "base": 768,
                    "large": 1024,
                    "giant": 1536} # tamaños del feature vector de cada version de dinov2

    backbone_size = 'small'
    with torch.no_grad():
        model = load_model(backbone_size)



    features, attention, grid_shape = get_dense_descriptor(model, img)
    # Primero, aplicas PCA a tus características originales para reducir la dimensionalidad
    pca = PCA(n_components=3)
    pca_trained = pca.fit(attention)



    image_filenames = os.listdir(dataset_path)
    

    #for img_filename in tqdm(image_filenames):

    # Ordena los nombres de archivo alfabéticamente
    sorted_image_filenames = sorted(image_filenames, key=lambda x: int(x.replace("page", "").replace(".jpg", "")))

    for img_filename in tqdm(sorted_image_filenames):
        torch.cuda.empty_cache()
        print(img_filename)
        image_path = os.path.join(dataset_path, img_filename)
        img = io.imread(image_path)
        features, attention, grid_shape = get_dense_descriptor(model, img)

        #fmap_shape = grid_shape + (features.shape[-1],)
        
        # visualizar mapas usando PCA
      
        attention_rgb_no_bg, inverted, remove = pca_colorize_2(attention, grid_shape, pca_trained)

        #attention_mask_box = foreground_mask_2(attention_rgb_no_bg, inverted, remove, use_bbox=True)
       


        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[1].imshow(attention_rgb_no_bg)

        # Save the figure
        plt.savefig(save_path+'/'+img_filename)  
        # Convertir antes de guardar
        #attention_rgb_no_bg = convert_image(attention_rgb_no_bg)
        #attention_mask_box = convert_image(attention_mask_box)

        # En tu bucle principal, antes de guardar la imagen:
        #scale_factor = 8
        #attention_rgb_no_bg = resize_image_pil(attention_rgb_no_bg, scale_factor)
        #ttention_mask_box = resize_image_pil(attention_mask_box, scale_factor)

        # Define las rutas completas de archivo, incluyendo el nombre del archivo y la extensión
        #save_path_rgb_no_bg = os.path.join(save_path_rgb, img_filename)
        #save_path_mask_box = os.path.join(save_path_mask, img_filename)

        # Guardar las imágenes
        #io.imsave(save_path_rgb_no_bg, attention_rgb_no_bg)
        #io.imsave(save_path_mask_box, attention_mask_box)

        del image_path
        del img
        del features
        del attention
        del grid_shape
        del attention_rgb_no_bg
        del inverted
        #del attention_mask_box
        torch.cuda.empty_cache()
