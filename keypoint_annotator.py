import os
import cv2
import shutil
import numpy as np

# Presiona 'a' o 'd' para navegar entre las etiquetas de una misma imagen.
# Presiona 'q' o 'e' para navegar entre las diferentes imágenes del directorio.
# Presiona 'click izquierdo' para añadir un keypoint.
# Presiona 'x' para añadir un punto no visible (-1, -1).
# Presiona 'b' para eliminar keypoints.
# Presiona 'n' para eliminar la imagen y sus etiquetas de la carpeta de imágenes.
# Presiona 'Esc' para abandonar el editor.

# Las anotaciones originales se almacenan en la carpeta 'temp' como backup. Cualquier cambio introducido en
# las anotaciones se sobreescribe automáticamente en la carpeta 'labels'.
# Presiona 'r' para recuperar un archivo del back-up.
# Presiona 's' para sobreescribir un archivo del back-up con las anotaciones actuales.

# Dónde se encuentran las imagenes y anotaciones
dataset_path = './dataset/'
images_path = f'{dataset_path}/images/'
labels_path = f'{dataset_path}/labels/'

temp_path = f'{dataset_path}/temp/'
if not os.path.exists(temp_path):
    os.makedirs(temp_path)

image_files = [file for file in os.listdir(images_path) if file.endswith(('.jpg', '.png', '.jpeg'))]
image_files.sort()

for image_filename in image_files:
    label_filename = '.'.join(image_filename.split('.')[0:-1]) + '.txt'
    if label_filename.endswith('.txt'):
        ruta_completa = os.path.join(labels_path, label_filename)
        if os.path.getsize(ruta_completa) == 0:
            # print(f"El archivo {label_filename} está vacío.")
            image_files.remove(image_filename)

def load_labels(image_filename):
    label_filename = '.'.join(image_filename.split('.')[0:-1]) + '.txt'
    if os.path.isfile(f'{labels_path}/{label_filename}'):
        if not os.path.isfile(f'{temp_path}/{label_filename}'):
            shutil.copy(f'{labels_path}/{label_filename}', f'{temp_path}/{label_filename}')
        with open(f'{labels_path}/{label_filename}', 'r') as file:
            return file.read().splitlines()
    else:
        return []


def save_labels(image_filename):
    label_filename = '.'.join(image_filename.split('.')[0:-1]) + '.txt'
    shutil.copy(f'{temp_path}/{label_filename}', f'{labels_path}/{label_filename}')
    pass

def draw_rectangle(img, current_label):
    label_parts = current_label.split()
    # class_id = int(label_parts[0])
    x, y, w, h = map(float, label_parts[1:5])

    # Dibujar el rectángulo en la imagen
    start_point = (int((x - w / 2) * img.shape[1]), int((y - h / 2) * img.shape[0]))
    end_point = (int((x + w / 2) * img.shape[1]), int((y + h / 2) * img.shape[0]))
    color = (0, 255, 0)  # Color verde
    thickness = 2
    img = cv2.rectangle(img, start_point, end_point, color, thickness)
    return img


def check_point_inside_rectangle(point, current_label):
    label_parts = current_label.split()
    # x_p, y_p = point
    x_p = point[0] / img.shape[1]
    y_p = point[1] / img.shape[0]
    x, y, w, h = map(float, label_parts[1:5])
    # print(type(x), y, w, h)

    half_width = w / 2
    half_height = h / 2

    if (x - half_width) <= x_p <= (x + half_width) and (y - half_height) <= y_p <= (y + half_height):
        return True
    else:
        return False


def get_keypoints(img, current_label):
    label_parts = current_label.split()
    points = []
    if len(label_parts) > 5:
        for idx in range(5, len(label_parts), 2):
            points.append((int(float(label_parts[idx])*img.shape[1]), int(float(label_parts[idx+1])*img.shape[0])))
    return points

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if check_point_inside_rectangle((x, y), current_label):
            points.append((x, y))
        else:
            print('The point is outside the box!')

def save_annots(labels_path, current_label_filename, labels, current_label_index, points):
    with open(f"{labels_path}/{current_label_filename}", 'w') as file:
        for label_idx, label_text in enumerate(labels):
            if label_idx == current_label_index:
                # Actualizar la etiqueta actual con el nuevo punto clave
                label_text = label_text.split()[0:5]
                for point in points:
                    label_text.extend([f'{point[0] / img.shape[1]}', f'{point[1] / img.shape[0]}'])
                label_text = ' '.join(label_text)
            file.write(label_text + '\n')
    # print(f"Etiqueta {current_label_filename} actualizada.")


# Inicializar variables
current_image_index = 0
current_label_index = 0

# Crear ventana de visualización de la imagen
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 1200, 1000)
cv2.setMouseCallback('Image', mouse_callback)

while True:
    current_image_filename = image_files[current_image_index]
    # print(current_image_filename)

    labels = load_labels(current_image_filename)
    while labels == []:
        image_files.remove(current_image_filename)
        current_image_filename = image_files[current_image_index]
        labels = load_labels(current_image_filename)
    # current_label_index = 0 if labels else None
    current_label = labels[current_label_index]
    current_label_filename = '.'.join(current_image_filename.split('.')[0:-1]) + '.txt'

    image_path = os.path.join(images_path, current_image_filename)
    img = cv2.imread(image_path)
    img_copy = img.copy()
    points = get_keypoints(img, current_label)

    if current_label != []:
        img = draw_rectangle(img, current_label)

    # Dibujar puntos clave en la imagen
    for point in points:
        img = cv2.circle(img, point, 4, (0, 0, 255), -1)  # Color rojo

    # Mostrar la imagen
    text = ' '.join([str(point) for point in points])
    cv2.putText(img, text, (30, 30), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Image', img)

    # Esperar a que el usuario presione una tecla
    key = cv2.waitKey(1)

    # Manejar las teclas presionadas
    if (key == ord('a') and current_label_index > 0) or (key == ord('d')): # and current_label_index < len(labels) - 1
        if key == ord('a'):
            current_label_index -= 1
        elif key == ord('d'):
            if current_label_index < len(labels) - 1:
                current_label_index += 1
            else:
                for label in labels:
                    img_copy = draw_rectangle(img_copy, label)
                    points = get_keypoints(img, label)
                    for point in points:
                        img_copy = cv2.circle(img_copy, point, 4, (0, 0, 255), -1)  # Color rojo

                winname = 'Labeled Image. Press e to go next, any other to return to this one.'
                cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(winname, 1200, 1000)
                cv2.imshow(winname, img_copy)
                key2 = cv2.waitKey(0)
                cv2.destroyWindow(winname)

                if key2 == ord('e'):
                    key = key2
                else:
                    current_label_index = 0

        current_label = labels[current_label_index]
        points = get_keypoints(img, current_label)

    elif key == ord('b') and points:
        # Borrar el último punto clave
        points.pop()
        # save_annots(labels_path, current_label_filename, labels, current_label_index, points)

    elif key == ord('x'):
        points.append((-1, -1))

    # Salir si el usuario presiona la tecla 'Esc'
    elif key == 27:
        # for filename in image_files[0:25]:
        #     shutil.copy(f'{images_path}/{filename}', f'{dataset_path}/test_dataset/images/train/{filename}')
        #     shutil.copy(f'{images_path}/{filename}', f'{dataset_path}/test_dataset/images/valid/{filename}')
        #
        #     label_filename = '.'.join(filename.split('.')[0:-1]) + '.txt'
        #     shutil.copy(f'{labels_path}/{label_filename}', f'{dataset_path}/test_dataset/labels/train/{label_filename}')
        #     shutil.copy(f'{labels_path}/{label_filename}', f'{dataset_path}/test_dataset/labels/valid/{label_filename}')

        break

    if key != ord('r'):
        # Guardar las modificaciones realizadas
        save_annots(labels_path, current_label_filename, labels, current_label_index, points)

    if (key == ord('q') and current_image_index > 0) or (key == ord('e') and current_image_index < len(image_files) - 1) or key == ord('r'): #

        if key == ord('q'):
            current_image_index -= 1
        elif key == ord('e'):
            current_image_index += 1
        elif key == ord('r'):
            shutil.copy(f'{temp_path}/{current_label_filename}', f'{labels_path}/{current_label_filename}')

        current_label_index = 0

    if key == ord('s'):
        shutil.copy(f'{labels_path}/{current_label_filename}', f'{temp_path}/{current_label_filename}')

    # save_annots(labels_path, current_label_filename, labels, current_label_index, points)

    if key == ord('n'):
        os.remove(f'{labels_path}/{current_label_filename}')
        os.remove(f'{images_path}/{current_image_filename}')
        image_files.remove(current_image_filename)
        current_label_index = 0


    # if key == ord('r'):
    #     shutil.copy(f'{temp_path}/{current_label_filename}', f'{labels_path}/{current_label_filename}')
    #     current_label_index = 0




# Cerrar la ventana y finalizar
cv2.destroyAllWindows()
