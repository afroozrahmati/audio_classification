from preprocessing import *
import glob
import numpy as np
import pickle
import random
from pydub import AudioSegment

#input parameters

def audio_segmentation(audio_file,segments,normal_abnormal_folder):
    filename = os.path.basename(audio_file)
    filename = filename.split('.')[0]
    rate, data = wavfile.read(audio_file)
    seq_len = data.shape[0] / float(rate)

    #divide if the audio lenght>3 seconds
    div = seq_len // segments

    t1=0
    t2=segments
    while div>0:
        t1 = t1 * 1000  # Works in milliseconds
        t2 = t2 * 1000
        newAudio = AudioSegment.from_wav(audio_file)
        newAudio = newAudio[t1:t2]
        if len(newAudio)>0:
            newAudio.export('./data/augmented_data/'+normal_abnormal_folder+'/'+filename+'_'+str(div)+'.wav', format="wav")  # Exports to a wav file in the current path.
        div-=1
        t1=t2/1000
        t2=t1+segments

def file_augmentation(segments):

    pathnormal = './data/pascal/normal/'  # './data/physionet/normal/'
    pathabnormal = './data/pascal/abnormal/'  # './data/physionet/abnormal/'

    files = glob.glob(pathnormal + '/*.wav')
    for file in files:
        audio_segmentation(file,segments,'normal')
        print(file)

    files = glob.glob(pathabnormal + '/*.wav')
    for file in files:
        audio_segmentation(file,segments,'abnormal')
        print(file)


#file_augmentation(5)



j = 0
i=0
for duration in [5]:
    for features in [40]:
        for dataset in ['physionet']:
            for timesteps in [128]:

                preprocess = preprocessing(duration=duration)

                pathnormal=  './data/'+dataset+'/normal/'  #'./data/physionet/normal/'
                pathabnormal ='./data/'+dataset+'/abnormal/'  #'./data/physionet/abnormal/'

                x_data_normal = []
                y_data_normal =[]
                y_data_normal =[]
                x_data_abnormal = []
                y_data_abnormal =[]

                hop_lenght = math.ceil((duration*22050) /timesteps)+1
                # hop_lenght =865 for duartion=5
                # hop_lenght = 520 for duration=3
                # hop_lenght = 1725 for duration =10
                # hop_lenght=0
                # if timesteps==64:
                #      hop_lenght=1740
                # #
                # if timesteps==256:
                #      hop_lenght=432
                #
                # if duration==10:
                #     hop_lenght=1725
                #hop_lenght=345
                files = glob.glob(pathnormal + '/*.wav')
                for file in files:
                    if i <50 :
                        x_data_normal.append(preprocess.extract_features(file,hop_lenght,features,timesteps))
                        y_data_normal.append([1,0])
                        print(file)
                        i+=1

                files = glob.glob(pathabnormal + '/*.wav')
                for file in files:
                    if j < 50:
                        x_data_abnormal.append(preprocess.extract_features(file,hop_lenght,features,timesteps))
                        y_data_abnormal.append([0,1])
                        print(file)
                        j+=1


                print("shape x normal data:", np.shape(x_data_normal) )
                print("shape y normal data:", np.shape(y_data_normal) )
                print("shape x abnormal data:", np.shape(x_data_abnormal)  )
                print("shape y abnormal data:", np.shape(y_data_abnormal)  )


                data_x = np.concatenate((x_data_normal,x_data_abnormal))
                data_y = np.concatenate((y_data_normal,y_data_abnormal))

                # Shuffle two lists with same order
                # Using zip() + * operator + shuffle()
                temp = list(zip(data_x, data_y))
                random.shuffle(temp)
                data_x, data_y = zip(*temp)


                print("shape xdata:", np.shape(data_x) )
                print("shape y data:", np.shape(data_y) )

                dir = './data/processed_files/'

                with open(dir+dataset+'_'+str(features)+'_'+str(duration)+'_'+str(timesteps)+'_test100_X.pkl', 'wb') as o:
                    pickle.dump(data_x, o, pickle.HIGHEST_PROTOCOL)

                with open(dir+dataset+'_'+str(features)+'_'+str(duration)+'_'+str(timesteps)+'_test100_Y.pkl', 'wb') as o:
                    pickle.dump(data_y, o, pickle.HIGHEST_PROTOCOL)
