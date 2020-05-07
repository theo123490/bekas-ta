import json
import pandas as pd
from img_listing import image_list



meta_list = pd.read_csv('metadata_mainan.csv')
idx = max(meta_list.index.tolist())
#meta_list.columns = meta_list.columns.str.replace('.','_')

img_list = image_list('Melanoma_json/0024258-0026989')

for i in img_list:     
     json_data = open('Melanoma_json/0024258-0026989/' + i[:-4] + '.json')
     data = json.load(json_data)
     _id = data['_id']
     name = data['name']
     
     meta = data['meta']
     
     meta_ac = meta['acquisition']
     
     image_type = meta_ac['image_type'] 
     pixelsX = meta_ac['pixelsX'] 
     pixelsY = meta_ac['pixelsY'] 
     
     
     
     meta_cl = meta['clinical']
     age_approx = meta_cl['age_approx']
     diagnosis = meta_cl['diagnosis']
     
     idx = idx + 1
     
     df2 = pd.DataFrame(data = {'_id' : [_id],
                                'name' : [name] , 
                                'meta.clinical.diagnosis' : [diagnosis] , 
                                }, index = [idx])
     
     
     
     meta_list = meta_list.append(df2)
     print('appending : ' + i )
     
meta_list.to_csv('metadata_mainan.csv')





