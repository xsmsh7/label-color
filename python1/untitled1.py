annotation_file = "blue helmet229.txt"

%cd ../labels

with open(annotation_file, "r") as file:
    annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(" ") for x in annotation_list]
    annotation_list = [[float(y) for y in x ] for x in annotation_list]
    
#Get the corresponding image file
image_file = annotation_file.replace(
    "annotations", "images").replace("txt", "jpg")
%cd ../images
assert os.path.exists(image_file)

#Load the image
image = Image.open(image_file)

#Plot the Bounding Box
label_table = f.plot_label(image, annotation_list)
#f.plot_bounding_box(image, annotation_list)

for i in range(0, len(label_table)):
    PIL_image = label_table[i][1]
    plt.imshow(PIL_image)
    plt.title(annotation_file)
    plt.show()
    PIL_image = remove(PIL_image)#,session = new_session(model_name = "u2net")
    
    
    PIL_image = np.array(PIL_image)
    PIL_image[PIL_image[:,:,3] < 60] = 0
    
    label_table[i].append(PIL_image)
    
    #rgb to hsv
    hsvim = f.rgb_to_hsv(label_table[i][2])     
    label_table[i].append(hsvim)
    
    
for i in range(0, len(label_table)):    
    #color kmean
    centers,number_weight = f.image_color_separate(label_table[i][3])
    
    label_table[i].append(centers)
    label_table[i].append(number_weight)


for i in range(0, len(label_table)):
    color_weight = []
    for j in range(0, 3):
        color = f.loss_function(goal, label_table[i][4][j, :])
        color_weight.append(color)
    label_table[i].append(np.array(color_weight))


for i in range(0, len(label_table)):
    title = "None" + annotation_file + str(label_table[i][0])
    
    if label_table[i][5][label_table[i][6] == 'blue'].sum() > 0.4:
        title = "Blue " + annotation_file + str(label_table[i][0])
        annotation_list[label_table[i][0]-1][0] = 3
        %cd ../newlabels0
        f.change_label(annotation_file, annotation_list)
    if label_table[i][5][label_table[i][6] == 'white'].sum() > 0.4:
        title = "White " + annotation_file + str(label_table[i][0])
        annotation_list[label_table[i][0]-1][0] = 4
        %cd ../newlabels0
        f.change_label(annotation_file, annotation_list)
    plt.imshow(label_table[i][2])
    plt.title(title)
    plt.show()