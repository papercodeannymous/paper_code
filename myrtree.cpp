/**
 * 
 * Extracts subtree-related states from an R-tree node and constructs corresponding vectors:
 *      OR2
 *      SDP
 *      LAR LPR
 */

// void RTree::GetEvaluatedInsertStates125(RTreeNode* tree_node, Rectangle* rec, double* states, int cb, int num_features) {
//     /**
//      * Among different state design combinations (1, 2, 3, 4, 5, …, ), 
//      * combination 1, 2, 5 (OR2, SDP, LR) achieves the best performance.
//      */
//     int state_num = num_features;
//     int dimension = cb * state_num;
//     for (int i = 0; i < dimension; i++) {
//         states[i] = 0;  
//     }
//     double max_state1 = -std::numeric_limits<double>::infinity();
//     double max_state2 = -std::numeric_limits<double>::infinity();
//     double max_state3 = -std::numeric_limits<double>::infinity();
//     double max_state4 = -std::numeric_limits<double>::infinity();

//     bool onLayerFlag = false;
//     for (int i = 0; i < comprehensive_evaluated_children.size(); i++) {
//         if (i == cb) break;
//         RTreeNode* node = comprehensive_evaluated_children[i];

//         // generate i-th evaluated child's state
//         int choosed_child = i * state_num;

//         /**OR2 */
//         double node_fill_factor = 0.0;
//         // ...
//         states[choosed_child] = node_fill_factor;

//         /**SDP */ 
//         double total_dist = 0.0;
//         RTreeNode* iter = node;
//         int origin_node_level = node->level;
//         std::list<RTreeNode*> queue;
//         queue.push_back(iter);
//         while (!queue.empty()) {
//             iter = queue.front();
//             queue.pop_front();

//             // modification
//             // if (node->level - iter->level >= 3) continue;

//             // ...
//         }
//         states[choosed_child + 1] = total_dist;

//         /* Here can handle other kind of states*/
//         double subtree_area = 0.0;
//         double subtree_perimeter = 0.0;
//         RTreeNode* sub_iter = node;
//         std::list<RTreeNode*> sub_queue;
//         sub_queue.push_back(sub_iter);

//         std::vector<double> layer_area(50, 0.0);
//         std::vector<double> layer_perimeter(50, 0.0);
//         int sub_node_cnt = 1;

//         /**LAR LPR */
        
//         while (!sub_queue.empty()) {
//             sub_iter = sub_queue.front();
//             sub_queue.pop_front();
//             sub_node_cnt += sub_iter->entry_num;

//             // modification
//             // if (node->level - sub_iter->level >= 3) continue;
            
//             if (sub_iter->is_leaf) {
//                 ...
//             } else {
//                 ...
//             }
//         }
//         subtree_area = std::accumulate(layer_area.begin(), layer_area.end(), 0.0);
//         subtree_perimeter = std::accumulate(layer_perimeter.begin(), layer_perimeter.end(), 0.0);

//         if (layer_area[1] == 0) {
//             onLayerFlag = true;
//             states[choosed_child + 2] = subtree_area / sub_node_cnt;
//             states[choosed_child + 3] = subtree_perimeter / sub_node_cnt;
//         } else {
//             std::vector<double> layer_area_ratio;
//             std::vector<double> layer_peri_ratio;
//             for (size_t i = 0; i < layer_area.size(); i++) {
//                 if (layer_area[i + 1] > 0.0) {
//                     // ratio calculation

//                 } else {
//                     break;
//                 }
//             }
//             states[choosed_child + 2] = std::accumulate(layer_area_ratio.begin(), layer_area_ratio.end(), 0.0);
//             states[choosed_child + 3] = std::accumulate(layer_peri_ratio.begin(), layer_peri_ratio.end(), 0.0);
//         }

//         // fill state 

//     // if current node has few children, even less than cb
//     // copy children states existed to fill the blank to get cb
//     if (comprehensive_evaluated_children.size() < cb) {
//         // ...
//     }

//     // Normalization
//     for (int i = 0; i < dimension; i++) {
//         ...
//     }
// }