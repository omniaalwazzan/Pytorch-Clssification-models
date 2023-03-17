class Dataset_(Dataset):
    def __init__(self, image_dir,df_meta,transform=None):
        self.image_dir = image_dir # 
        self.df_meta=df_meta # --> csv file
        self.LABEL = df_meta['classification']
        self.images = df_meta['Image_name']


        self.transform = transform

    def __len__(self):
        return self.df_meta.shape[0]
    
    def __getitem__(self, index):
        ## reading image ###
        img_path = os.path.join(self.image_dir, self.images[index])
        image = cv2.imread(img_path)
        image_name=self.images[index]
        gt = self.LABEL[index]
        
        c1=self.df_meta.loc[self.df_meta['Image_name']==image_name]

        c2 = c1.iloc[:, 2:-3]
        c2 =np.array(c2)
        


        #cLin_features = np.array(cLin_features)
        if self.transform is not None:
            image = self.transform(image)

            
        return image,gt,self.images[index]
            

## Read the data ##
dataset = Dataset_(image_dir=dir_,df_meta=meta_df,transform=transform)
#train_loader = torch.utils.data.DataLoader(load_data,batch_size=30)



for batch_i, data in enumerate(train_loader):

    # extract data
    inputs = data[0]
    labels = data[1]
    img_name = data[2]
    print(inputs.shape)
    print(labels.shape)
    # create plot
    fig = plt.figure(figsize = (14, 7))
    for i in range(8):
        ax = fig.add_subplot(2,6, i + 1, xticks = [], yticks = [])
        #ax = fig.add_subplot(2,1,i ,xticks = [], yticks = [])     

        plt.imshow(inputs[i].numpy().transpose(1, 2, 0))
        ax.set_xlabel(img_name[i])
        ax.set_title(labels.numpy()[i])
        

    break
