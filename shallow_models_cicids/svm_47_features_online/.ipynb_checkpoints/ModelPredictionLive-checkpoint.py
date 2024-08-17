###########################################################################################

from nfstream import NFPlugin, NFStreamer
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

with open('C:/Users/katsa/OneDrive/Jupyter_files/make_http_requests/urls.txt', 'r') as f:
    url_list = [line.strip() for line in f.readlines()]

selected_features_nfstream = ['dst_port', 'src2dst_packets', 'dst2src_packets',
'src2dst_bytes', 'dst2src_bytes', 'src2dst_max_ps', 'src2dst_min_ps', 'src2dst_mean_ps',
'src2dst_stddev_ps', 'dst2src_max_ps', 'dst2src_min_ps', 'dst2src_mean_ps', 
'dst2src_stddev_ps', 'bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms',
'bidirectional_max_piat_ms', 'bidirectional_min_piat_ms', 'src2dst_mean_piat_ms', 
'src2dst_stddev_piat_ms', 'src2dst_max_piat_ms', 'src2dst_min_piat_ms', 'dst2src_mean_piat_ms', 
'dst2src_stddev_piat_ms', 'dst2src_max_piat_ms', 'dst2src_min_piat_ms', 'bidirectional_min_ps', 
'bidirectional_max_ps', 'bidirectional_mean_ps', 'bidirectional_stddev_ps', 
'bidirectional_fin_packets', 'bidirectional_syn_packets', 'bidirectional_rst_packets', 
'bidirectional_psh_packets', 'bidirectional_ack_packets', 'bidirectional_urg_packets',
'bidirectional_cwr_packets', 'bidirectional_ece_packets']

added_features_nfstream = ['src2dst_psh_packets', 'src2dst_urg_packets', 
                           'dst2src_psh_packets', 'dst2src_urg_packets']
# these features can only be known by the person who set those values on nfstream
#extra_features = ['active_timeout', 'idle_timeout']

# features that need to be computed from nfstream primitives manually
# added at the end of the feature list so that there's no need to 
# rearrange the order of feature columns in x_test
computed_features_nfstream = ['bidirectional_duration_ms', 'flow_bytes/s', 
                              'flow_packets/s', 'fwd_packets/s', 'bwd_packets/s', 
                              'packet_length_variance']
selected_features_nfstream = selected_features_nfstream + added_features_nfstream + computed_features_nfstream


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
        # convert duration milliseconds to microseconds
        flow_df['bidirectional_duration_ms'] = flow_df['bidirectional_duration_ms'] * 10**3
        flow_df.rename(columns={'bidirectional_duration_ms':'bidirectional_duration_us'}, inplace=True)
        # do the same for all interarrival time features
        change_time_scale_list = ['bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms', 
                                  'bidirectional_max_piat_ms', 'bidirectional_min_piat_ms', 
                                  'src2dst_mean_piat_ms', 'src2dst_stddev_piat_ms', 
                                  'src2dst_max_piat_ms', 'src2dst_min_piat_ms', 
                                  'dst2src_mean_piat_ms', 'dst2src_stddev_piat_ms', 
                                  'dst2src_max_piat_ms', 'dst2src_min_piat_ms'] 
        for feature_ms in change_time_scale_list:
            flow_df[feature_ms] = flow_df[feature_ms] * 10**(3)
            tmp_feature_us = str(feature_ms).removesuffix('ms') + 'us'
            flow_df.rename(columns={str(feature_ms):tmp_feature_us}, inplace=True)
        # add composite features
        flow_duration_seconds = flow_df['bidirectional_duration_us'] / 10**6
        flow_df['flow_bytes/s'] = flow.bidirectional_bytes / flow_duration_seconds
        flow_df['flow_packets/s'] = flow.bidirectional_packets / flow_duration_seconds
        flow_df['fwd_packets/s'] = flow_df['src2dst_packets'] / flow_duration_seconds
        flow_df['bwd_packets/s'] = flow_df['dst2src_packets'] / flow_duration_seconds
        flow_df['packet_length_variance'] = flow_df['bidirectional_stddev_ps']**2
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