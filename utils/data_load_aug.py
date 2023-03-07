

## Data Analysis ##
train = pd.read_csv(meta_csv)
train.columns
bening_ = train.loc[train['classification'] == 'Benign', 'Number of Images'].sum()
Malignant_ = train.loc[train['classification'] == 'Malignant', 'Number of Images'].sum()
print(f'Nr. of Benign MRI (per-patient) in all dataset: {bening_}')
print(f'Nr. of Malignant MRI (per-patient) in all dataset: {Malignant_}')
train = train.drop(columns=['Number of Images','number'])
train.columns
# age impute
train.iloc[:,3]= (train.iloc[:, 3]- train.iloc[:, 3].mean()) / train.iloc[:, 3].std()
# Encode the categorical variable abnormality and LeftRight
le = LabelEncoder()
train['abnormality'] = le.fit_transform(train['abnormality'])
train['LeftRight'] = le.fit_transform(train['LeftRight'])
# change the class label to 0 or 1
train['classification'] = train['classification'].apply(lambda x: 0 if x == 'Benign' else 1)
train.classification.value_counts()

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img
size = 96
s =1  
color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * size)
data_transforms = transforms.Compose([#transforms.RandomResizedCrop(size=size),
                                        transforms.ToPILImage(),
                                        transforms.RandomHorizontalFlip(),
                                        #transforms.RandomApply([color_jitter], p=0.8),
                                        GaussianBlur(kernel_size=int(0.1 * size)),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.ToTensor()])   



class Dataset_(Dataset):
    def __init__(self, image_dir,df_meta,transform=None):
        self.image_dir = image_dir # --> r'E:/IAAA_CMMD/manifest-1616439774456/CMMD/'
        
        self.df_meta=df_meta # --> csv file
        self.LABEL = df_meta['classification']       
        self.images = df_meta['Image_name']
        self.transform = transform

    def __len__(self):
        return self.df_meta.shape[0]
    
    def __getitem__(self, index):
        ## reading image ###
        img_path = os.path.join(self.image_dir, self.images[index])
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_three = cv2.merge([gray,gray,gray])
        img__ = cv2.resize(gray_three, (96,96), interpolation = cv2.INTER_AREA)
        #img__ = img__.transpose([2,0,1])
        image_name=self.images[index]
        gt = self.LABEL[index]
        
        c1=self.df_meta.loc[self.df_meta['Image_name']==image_name]

        c2 = c1.iloc[:, 2:-3]
        c2 =np.array(c2)
        
        if self.transform is not None:
            a = self.transform(img__)#.transpose([2,0,1]
                                                

            
        return img__,c2,gt,self.images[index]

## Read the data ##
load_data = Dataset_(image_dir=dir_,df_meta=train,transform=data_transforms)



## Data Spliting ##

# Define the sizes of the training, validation, and testing sets
train_size = int(0.7 * len(load_data))
val_size = int(0.15 * len(load_data))
test_size = len(load_data) - train_size - val_size

# Use random_split to split the dataset into training, validation, and testing sets
train_set, val_set, test_set = random_split(load_data, [train_size, val_size, test_size])

gt_t = train_set.dataset.LABEL[train_set.indices]

gt_v = val_set.dataset.LABEL[val_set.indices]

gt_te = test_set.dataset.LABEL[test_set.indices]


## Data Generator ##
def fusion_dg():
    train_loader = DataLoader(train_set, batch_size=batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,shuffle=False)
    
    print('Len. of train data', len(train_loader.dataset))
    print('Len. of valid data', len(val_loader.dataset))
    print('Len. of test data', len(test_loader.dataset))
    
    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = fusion_dg()
