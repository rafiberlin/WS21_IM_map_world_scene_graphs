import argparse
import os
import json

import numpy as np

import torch
from tqdm import tqdm
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog

def preprocess_scene_graphs_output( detected_path, output_file_name):
    data_dir = DatasetCatalog.DATA_DIR
    attrs = DatasetCatalog.DATASETS["VG_stanford_filtered_with_attribute"]
    cap_graph_file = os.path.join(data_dir, attrs["capgraphs_file"])
    vg_dict_file = os.path.join(data_dir, attrs["dict_file"])
    image_file = os.path.join(data_dir, attrs["image_file"])
    output_path = os.path.join(detected_path, output_file_name)


    cap_graph = json.load(open(cap_graph_file))

    vg_dict = json.load(open(vg_dict_file))
    vg_info = json.load(open(image_file))

    # generate union predicate vocabulary
    sgg_rel_vocab = list(set(cap_graph['idx_to_meta_predicate'].values()))
    sgg_rel2id = {key: i+1 for i, key in enumerate(sgg_rel_vocab)}



    # generate union object vocabulary
    sgg_obj_vocab = list(set(vg_dict['idx_to_label'].values()))
    sgg_obj2id = {key: i+1 for i, key in enumerate(sgg_obj_vocab)}




    # generate scene graph from test results
    def generate_detect_sg(predictions, det_info, obj_thres = 0.1):
        num_img = len(predictions)


        output = {}
        for i in tqdm(range(num_img)):
            # key = det_info['idx_to_files'][i]
            # i = str(i)
            # all_obj_labels = predictions[i]['pred_labels']
            # all_obj_scores = predictions[i]['pred_scores']
            # all_rel_pairs = predictions[i]['rel_pair_idxs']
            # all_rel_prob = predictions[i]['pred_rel_scores']
            all_obj_labels = predictions[i].get_field('pred_labels')
            all_obj_scores = predictions[i].get_field('pred_scores')
            all_rel_pairs = predictions[i].get_field('rel_pair_idxs')
            all_rel_prob = predictions[i].get_field('pred_rel_scores')
            all_rel_scores, all_rel_labels = all_rel_prob.max(-1)

            # filter objects and relationships
            all_obj_scores[all_obj_scores < obj_thres] = 0.0
            obj_mask = all_obj_scores >= obj_thres
            triplet_score = all_obj_scores[all_rel_pairs[:, 0]] * all_obj_scores[all_rel_pairs[:, 1]] * all_rel_scores
            rel_mask = ((all_rel_labels > 0) + (triplet_score > 0)) > 0

            # generate filterred result
            num_obj = obj_mask.shape[0]
            num_rel = rel_mask.shape[0]
            rel_matrix = torch.zeros((num_obj, num_obj))
            for k in range(num_rel):
                if rel_mask[k]:
                    rel_matrix[int(all_rel_pairs[k, 0]), int(all_rel_pairs[k, 1])] = all_rel_labels[k]
            rel_matrix = rel_matrix[obj_mask][:, obj_mask].long()
            filter_obj = all_obj_labels[obj_mask]
            filter_pair = torch.nonzero(rel_matrix > 0)
            filter_rel = rel_matrix[filter_pair[:, 0], filter_pair[:, 1]]

            # generate labels
            pred_objs = [vg_dict['idx_to_label'][str(i)] for i in filter_obj.tolist()]
            pred_rels = [[i[0], i[1], cap_graph['idx_to_meta_predicate'][str(j)]] for i, j in zip(filter_pair.tolist(), filter_rel.tolist())]
            file_name = det_info[i]["img_file"]
            output[file_name] = [{'entities' : pred_objs, 'relations' : pred_rels}, ]
        return output


    def generate_txt_img_sg(img_sg):
        txt_img_sg = {}

        for img in tqdm(img_sg.keys()):
                encode_img = {'entities':[], 'relations':[]}
                for item in img_sg[img]:
                    entities = [sgg_obj2id[e] for e in item['entities']]
                    relations = [[entities[r[0]], entities[r[1]], sgg_rel2id[r[2]]] for r in item['relations']]
                    encode_img['entities'] = encode_img['entities'] + entities
                    encode_img['relations'] = encode_img['relations'] + relations

                # ===================================================================================
                # ============================== Acknowledgement ====================================
                # ===================================================================================
                #     Since I lost part of the code when I merged several jupyter notes into this
                # preprocessing.py files, the "image_graph" and "text_graph" are missing in the
                # original preprocessing.py. Thanks to the Haeyong Kang from KAIST, he filled in
                # the missing part by the following code.
                # ===================================================================================

                # === for image_graph ============================================here
                entities = encode_img['entities']
                relations = encode_img['relations']
                if len(relations) == 0:
                    img_graph = np.zeros((len(entities), 1))
                else:
                    img_graph = np.zeros((len(entities), len(relations)))

                image_graph = []
                for i, es in enumerate(entities):
                    for j, rs in enumerate(relations):
                        if es in rs:
                            img_graph[i,j] = 1
                        else:
                            img_graph[i,j] = 0

                image_graph.append(img_graph.tolist())


                #txt_img_sg[coco_id] = {'img':encode_img, 'txt':encode_txt}
                txt_img_sg[img] = {
                    'img':encode_img,
                    'image_graph':image_graph
                }

        return txt_img_sg


    def img_coco_mapping():
        img_coco_map = {}
        for img_id, coco_id in zip(cap_graph['vg_image_ids'], cap_graph['vg_coco_ids']):
            img_coco_map[int(img_id)] = int(coco_id)
        return img_coco_map


    info_file_name = "custom_data_info.json"
    prediction_file_name = "custom_prediction.json"

    # detected_result = json.load(open(detected_path + prediction_file_name))
    # detected_info = json.load(open(detected_path + info_file_name))

    detected_result = torch.load(os.path.join(detected_path , "eval_results.pytorch"))
    detected_info = json.load(open(os.path.join(detected_path ,  "visual_info.json")))

    output = generate_detect_sg(detected_result["predictions"], detected_info, obj_thres = 0.1)

    # You can replace cap_graph['vg_coco_id_to_capgraphs'] by the graphs produced in
    # maskrcnn_benchmark/image_retrieval/sentence_to_graph_processing.py if you want.
    txt_img_sg = generate_txt_img_sg(output)

    with open(output_path, 'w') as outfile:
        json.dump(txt_img_sg, outfile)
    print("Output file created", output_path)


# Execute this file with two paratmeters:
# --test-results-path is the path to the results file produced tools/relation_test_net.py (contains eval_results.pytorch and visual_info.json)
# --output-file-name is the name of the output file (will be created under the path given for --test-results-path )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing of Scene Graphs for Image Retrieval")
    parser.add_argument(
        "--test_results_path",
        default=f"/home/users/alatif/data/ImageCorpora/vg/sgg_ade20k_output",
        help="path to config file",
    )

    parser.add_argument(
        "--output_file_name",
        default=f"sg_of_causal_sgdet_custom_img_graph_only.json",
        help="creates this file under the path specified with  --test-results-path",
    )

    args = parser.parse_args()

    # path = "/home/rafi/checkpoints/sgdet/"
    # outfile_name = "sg_of_causal_sgdet_custom_img_graph_only.json"
    preprocess_scene_graphs_output(args.test_results_path, args.output_file_name)
