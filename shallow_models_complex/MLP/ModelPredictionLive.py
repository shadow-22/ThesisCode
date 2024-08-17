###########################################################################################

from nfstream import NFPlugin, NFStreamer
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

with open('C:/Users/katsa/OneDrive/Jupyter_files/make_http_requests/urls.txt', 'r') as f:
    url_list = [line.strip() for line in f.readlines()]


selected_features_nb15 = ['dsport', 'dur', 'Spkts', 'Dpkts', 'sbytes', 'dbytes', 'smeanz', 
                          'dmeanz', 'flow_bytes/s', 'flow_packets/s', 'fwd_packets/s', 'bwd_packets/s']

selected_features_nfstream = ['dst_port', 'bidirectional_duration_ms', 'src2dst_packets', 
                              'dst2src_packets', 'src2dst_bytes', 'dst2src_bytes', 'src2dst_mean_ps', 
                              'dst2src_mean_ps', 'flows_bytes/s', 'flow_packets/s', 'fwd_packets/s', 'bwd_packets/s']


flow_attr_lst = []
feature_lst_names = []

class ModelPrediction(NFPlugin): 
    def preprocess(self, flow):
        flow_attr_lst = dir(flow)
        feature_lst = []
        feature_lst_names = []
        for feature in selected_features_nfstream:
            for attribute in flow_attr_lst:
                if not (attribute.startswith('__') or attribute.startswith('_')):
                    if (attribute == feature):
                        feature_lst.append(getattr(flow, attribute))
                        feature_lst_names.append(attribute)
        flow_df = pd.DataFrame(np.array([feature_lst]), columns=np.array(feature_lst_names))
        # convert the duration of the live flow from milliseconds to microseconds
        # it is assumed that the 'dur' flow duration feature of NB15 is in milliseconds
        # in the preprocessing it was multiplied by 10**3, so converted into microseconds
        # so we're doing the same here
        flow_df['bidirectional_duration_ms'] = flow_df['bidirectional_duration_ms'] * 10**3
        flow_df.rename(columns={'bidirectional_duration_ms':'bidirectional_duration_us'}, inplace=True)
        
        # if 'dur' is in seconds then it was saved as kilo seconds (kSec) during preprocessing
        # in that case composite features have to be divided by milli seconds...
        #flow_duration_mseconds = flow_df['bidirectional_duration_ms']
        #flow_duration_seconds = flow_df['bidirectional_duration_ms'] / 10**3
        #flow_df['bidirectional_duration_ms'] = flow_duration_seconds * 10**3
        #flow_df.rename(columns={'bidirectional_duration_ms' : 'bidirectional_duration_ks'}, inplace=True)
        
        # add composite features
        flow_duration_seconds = flow_df['bidirectional_duration_us'] / 10**6
        flow_df['flow_bytes/s'] = flow.bidirectional_bytes / flow_duration_seconds
        flow_df['flow_packets/s'] = flow.bidirectional_packets / flow_duration_seconds 
        flow_df['fwd_packets/s'] = flow_df['src2dst_packets'] / flow_duration_seconds 
        flow_df['bwd_packets/s'] = flow_df['dst2src_packets'] / flow_duration_seconds 
        # remove bad values
        # how to make flow.id work???
        #print('First ' + 'FlowID: ' + str(flow.id) + " shape-> " + str(flow_df.shape))
        flow_df = flow_df.replace([np.inf, -np.inf], np.nan)
        #col_all_nan = flow_df.columns[flow_df.isnull().all(0)]
        #print(col_all_nan)
        # some features get all nan and are removed by simpleimputer by default
        # the modified shape breaks the pipeline...
        # the keep_empty_features parameter is only available from version 1.2 of sklearn
        # the model and scaler also have to be trained with the same version
        # otherwise an error occrus with SimpleImputer not recognizing the 
        # kee_empty_features parameters
        simp = SimpleImputer(keep_empty_features=True)
        flow_df = simp.fit_transform(flow_df)
        #print('Second ' + 'FlowID: ' + str(flow.id) + " shape-> " + str(flow_df.shape))
        flow_df = self.my_scaler.transform(flow_df)
        return flow_df
    
    def on_init(self, packet, flow):
        flow.udps.model_prediction = 0
    def on_expire(self, flow):
        if flow.requested_server_name in url_list:
            proc_flow = self.preprocess(flow)
            flow.udps.model_prediction = self.my_model.predict(proc_flow)

            