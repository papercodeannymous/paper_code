import sys
import ctypes
import numpy as np
import random
from ctypes import CDLL, c_int, c_double, c_void_p
import os

# 获取当前文件所在目录的父目录（项目根目录）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 构建库文件路径
LIB_PATH = os.path.join(PROJECT_ROOT, 'cpp_out', 'libmytree.so')

# 构建测试数据路径
generated_data_DIR = os.path.join(PROJECT_ROOT, 'generated_data')

class RTree:
    def __init__(self, max_entry, min_entry_factor):
        # 检查库文件是否存在
        if not os.path.exists(LIB_PATH):
            raise FileNotFoundError(
                f"Library not found at {LIB_PATH}\n"
                f"Please run './compile.sh' first to build the library."
            )
        
        self.lib = CDLL(LIB_PATH)

        # needed C funcs
        self.lib.ConstructTree.argtypes = [c_int, c_int]
        self.lib.ConstructTree.restype = c_void_p

        self.lib.SetDefaultInsertStrategy.argtypes = [c_void_p, c_int]
        self.lib.SetDefaultInsertStrategy.restype = c_void_p

        self.lib.SetDefaultSplitStrategy.argtypes = [c_void_p, c_int]
        self.lib.SetDefaultSplitStrategy.restype = c_void_p

        self.lib.InsertRect.argtypes = [c_void_p, c_double, c_double, c_double, c_double]
        self.lib.InsertRect.restype = c_void_p

        self.lib.GetRoot.argtypes = [c_void_p]
        self.lib.GetRoot.restype = c_void_p

        self.lib.DefaultInsert.argtypes = [c_void_p, c_void_p]
        self.lib.DefaultInsert.restype = c_void_p

        self.lib.DefaultSplit.argtypes = [c_void_p, c_void_p]
        self.lib.DefaultSplit.restype = c_void_p

        self.lib.Clear.argtypes = [c_void_p]
        self.lib.Clear.restype = c_void_p

        self.lib.IsLeaf.argtypes = [c_void_p]
        self.lib.IsLeaf.restype = c_int

        self.lib.IsRoot.argtypes = [c_void_p]
        self.lib.IsRoot.restype = c_int

        self.lib.NodeEntries.argtypes = [c_void_p]
        self.lib.NodeEntries.restype = c_int

        self.lib.NodeID.argtypes = [c_void_p]
        self.lib.NodeID.restype = c_int

        self.lib.RetrieveSortedInsertStates.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, ctypes.POINTER(c_double)]
        self.lib.RetrieveSortedInsertStates.restype = c_void_p

        self.lib.GetMinAreaContainingChild.argtypes = [c_void_p, c_void_p, c_void_p]
        self.lib.GetMinAreaContainingChild.restype = c_int

        self.lib.InsertWithLoc.argtypes = [c_void_p, c_void_p, c_int, c_void_p]
        self.lib.InsertWithLoc.restype = c_void_p

        self.lib.InsertWithSortedLoc.argtypes = [c_void_p, c_void_p, c_int, c_void_p]
        self.lib.InsertWithSortedLoc.restype = c_void_p

        self.lib.QueryRectangle.argtypes = [c_void_p, c_double, c_double, c_double, c_double]
        self.lib.QueryRectangle.restype = c_int

        self.lib.TreeHeight.argtypes = [c_void_p]
        self.lib.TreeHeight.restype = c_int

        self.lib.CopyTree.argtypes = [c_void_p, c_void_p]
        self.lib.CopyTree.restype = c_void_p

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.lib.ActionToNodePtr.argtypes = [c_void_p, c_void_p, c_int]
        self.lib.ActionToNodePtr.restype = c_void_p

        self.lib.GetNumNodes.argtypes = [c_void_p]
        self.lib.GetNumNodes.restype = c_int

        self.lib.GetNumObjects.argtypes = [c_void_p]
        self.lib.GetNumObjects.restype = c_int

        self.lib.GetTotalEntries.argtypes = [c_void_p]
        self.lib.GetTotalEntries.restype = c_double

        self.lib.MakeRect.argtypes = [c_double, c_double, c_double, c_double]
        self.lib.MakeRect.restype = c_void_p

        self.lib.TraverseTree.argtypes = [c_void_p]
        self.lib.TraverseTree.restype = c_void_p

        self.lib.DirectInsert.argtypes = [c_void_p, c_void_p]
        self.lib.DirectInsert.restype = c_void_p

        self.lib.RRInsert.argtypes = [c_void_p, c_void_p]
        self.lib.RRInsert.restype = c_void_p

        self.lib.RRSplit.argtypes = [c_void_p, c_void_p]
        self.lib.RRSplit.restype = c_void_p

        self.lib.DirectSplitWithReinsert.argtypes = [c_void_p, c_void_p]
        self.lib.DirectSplitWithReinsert.restype = c_void_p

        self.lib.KNNQuery.argtypes = [c_void_p, c_double, c_double, c_int]
        self.lib.KNNQuery.restype = c_int

        self.lib.InsertWithEvaluatedLoc.argtypes = [c_void_p, c_void_p, c_int, c_void_p]
        self.lib.InsertWithEvaluatedLoc.restype = c_void_p

        self.lib.RetrieveEvaluatedInsertStates.argtypes = [c_void_p, c_void_p, c_void_p, c_int, ctypes.POINTER(c_double)]
        self.lib.RetrieveEvaluatedInsertStates.restype = c_void_p

        self.lib.RetrieveEvaluatedInsertStatesByType.argtypes = [c_void_p, c_void_p, c_void_p, c_int, ctypes.POINTER(c_double), c_int, c_int]
        self.lib.RetrieveEvaluatedInsertStatesByType.restype = c_void_p

        self.lib.RetrieveInsertStatesByTypeMCTSVersion.argtypes = [c_void_p, c_void_p, c_void_p, c_int, ctypes.POINTER(c_double), c_int, c_int]
        self.lib.RetrieveInsertStatesByTypeMCTSVersion.restype = c_void_p

        # attrs
        
        self.strategy_map = {
            "INS_AREA":0, "INS_MARGIN":1, "INS_OVERLAP":2, "INS_RSTAR":3, 
            "SPL_MIN_AREA":0, "SPL_MIN_MARGIN":1, "SPL_MIN_OVERLAP":2, "SPL_QUADRATIC":3
        }
        self.insert_strategy = None
        self.split_strategy = None
        self.max_entry = max_entry
        self.min_entry_factor = min_entry_factor
        self.tree_ptr = self.lib.ConstructTree(int(max_entry), int(self.max_entry * self.min_entry_factor))

        # attrs set at other place
        """
        self.rec_ptr
        self.node_ptr
        self.next_node_ptr
        """

    def SetDefaultInsertStrategy(self, strategy):
        self.insert_strategy = strategy
        self.lib.SetDefaultInsertStrategy(self.tree_ptr, self.strategy_map[strategy])
    
    def SetDefaultSplitStrategy(self, strategy):
        self.split_strategy = strategy
        self.lib.SetDefaultSplitStrategy(self.tree_ptr, self.strategy_map[strategy])
    
    def PrepareRectangle(self, ll_x, ll_y, tr_x, tr_y):
        """ 
        Insert incoming rectangle data into tree object buffer

        get the rectangle and tree root node pointer
        """
        self.rec_ptr = self.lib.InsertRect(self.tree_ptr, ll_x, ll_y, tr_x, tr_y)
        self.node_ptr = self.lib.GetRoot(self.tree_ptr)

    def DefaultInsert(self, ll_x, ll_y, tr_x, tr_y):
        self.PrepareRectangle(ll_x, ll_y, tr_x, tr_y)
        self.lib.DefaultInsert(self.tree_ptr, self.rec_ptr)
    
    def DefaultSplit(self):
        self.lib.DefaultSplit(self.tree_ptr, self.node_ptr)
    
    def Clear(self):
        print("Clear tree")
        self.lib.Clear(self.tree_ptr)

    def RetrieveSortedInsertStates(self, action_space, rl_type):
        """
            action_space = 5 or 10
            rl_type=0: RL for enlarged children and deterministic for non-enlarged children.
            rl_type=1: RL for non-enlarged children and deterministic for enlarged children.
        """
        if self.lib.IsLeaf(self.node_ptr):
            return None
        state_length = 4 * action_space         # action_space is topk, action_spcae_size
        state_c = (c_double * state_length)()
        self.lib.RetrieveSortedInsertStates(self.tree_ptr, self.node_ptr, self.rec_ptr, action_space, rl_type, state_c)
        states = np.ctypeslib.as_array(state_c)
        return states
    
    def GetMinAreaContainingChild(self):
        if self.lib.IsLeaf(self.node_ptr):
            return None
        child = self.lib.GetMinAreaContainingChild(self.tree_ptr, self.node_ptr, self.rec_ptr)
        if child < 0:
            return None
        else:
            # print(f"\nchild_id: {child}\n")
            return child
        
    def InsertWithLoc(self, loc):
        self.next_node_ptr = self.lib.InsertWithLoc(self.tree_ptr, self.node_ptr, loc, self.rec_ptr)
        # if current node is leaf, get the terminal state
        # else circularly update current node-ptr to the next node-ptr 
        # to enter next level to choose child to insert rectangle
        if self.lib.IsLeaf(self.node_ptr):
            return True
        else:
            self.node_ptr = self.next_node_ptr
            return False
    
    def InsertWithSortedLoc(self, loc):
        self.next_node_ptr = self.lib.InsertWithSortedLoc(self.tree_ptr, self.node_ptr, loc, self.rec_ptr)
        if self.lib.IsLeaf(self.node_ptr):
            return True
        else:
            self.node_ptr = self.next_node_ptr
            return False
        
    def Query(self, boundary):
        node_access = self.lib.QueryRectangle(self.tree_ptr, boundary[0], boundary[1], boundary[2], boundary[3])
        return node_access

    def AccessRate(self, boundary):
        """
        计算查询访问率（归一化）
        
        Args:
            boundary: 查询矩形 [ll_x, ll_y, tr_x, tr_y]
        
        Returns:
            访问率 = 访问的节点数 / 树的总节点数
            范围: [0, 1]，值越小表示查询效率越高
            
        Note:
            此定义衡量"查询浪费了多少存储空间"。
            如果希望衡量"每层平均访问节点数"，可改为:
                return 1.0 * node_access / self.GetTreeHeight()
        """
        node_access = self.lib.QueryRectangle(self.tree_ptr, boundary[0], boundary[1], boundary[2], boundary[3])
        # total_nodes = self.GetNumNodes()
        
        # if total_nodes == 0:
        #     return 0.0
        
        # return 1.0 * node_access / total_nodes
        
        height = self.lib.TreeHeight(self.tree_ptr)
        if height == 0:
            print("height is 0")
            input()
        return 1.0 * node_access / height
    
    def CopyTree(self, tree):
        self.lib.CopyTree(self.tree_ptr, tree)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def GetTreeHeight(self):
        return self.lib.TreeHeight(self.tree_ptr)
    
    def GetRoot(self):
        return self.lib.GetRoot(self.tree_ptr)
    
    def GetCurrentNode(self):
        """获取当前节点指针（用于插入过程中的迭代）"""
        # node_ptr 是在 PrepareRectangle 和 InsertWithLoc 等方法中更新的内部状态
        # 这里直接返回它，因为它是通过 C++ 函数调用设置的
        return self.node_ptr
    
    def GetTreePtr(self):
        """获取树的指针（用于跨树操作如 CopyTree）"""
        return self.tree_ptr

    def MakeRect(self, ll_x, ll_y, tr_x, tr_y):
        return self.lib.MakeRect(ll_x, ll_y, tr_x, tr_y)

    def IsRoot(self, node):
        return self.lib.IsRoot(node)
    
    def NodeEntries(self, node):
        return self.lib.NodeEntries(node)

    def NodeID(self, node):
        return self.lib.NodeID(node)
    
    def IsLeaf(self, node):
        return self.lib.IsLeaf(node)
    
    def TraverseTreeInfo(self):
        self.lib.TraverseTree(self.tree_ptr)   

    def DirectInsert(self, ll_x, ll_y, tr_x, tr_y):
        self.PrepareRectangle(ll_x, ll_y, tr_x, tr_y)
        self.node_ptr = self.lib.DirectInsert(self.tree_ptr, self.rec_ptr)
    
    def DirectRRInsert(self, ll_x, ll_y, tr_x, tr_y):
        self.PrepareRectangle(ll_x, ll_y, tr_x, tr_y)
        self.node_ptr = self.lib.RRInsert(self.tree_ptr, self.rec_ptr)

    def DirectRRSplit(self):
        self.lib.RRSplit(self.tree_ptr, self.node_ptr)
    
    def DirectSplitWithReinsert(self):
        self.lib.DirectSplitWithReinsert(self.tree_ptr, self.node_ptr)

    def KNNQuery(self, x, y, k):
        node_access = self.lib.KNNQuery(self.tree_ptr, x, y, int(k))
        return node_access

    def InsertWithEvaluatedLoc(self, action):
        self.next_node_ptr = self.lib.InsertWithEvaluatedLoc(self.tree_ptr, self.node_ptr, action, self.rec_ptr)
        if self.lib.IsLeaf(self.node_ptr):
            return True
        else:
            self.node_ptr = self.next_node_ptr
            return False
        
    def RetrieveEvaluatedInsertStates(self, action_space):
        if self.lib.IsLeaf(self.node_ptr):
            return None
        state_length = 8 * action_space         # action_space is topk, action_spcae_size
        state_c = (c_double * state_length)()
        self.lib.RetrieveEvaluatedInsertStates(self.tree_ptr, self.node_ptr, self.rec_ptr, action_space, state_c)
        states = np.ctypeslib.as_array(state_c)
        return states
    
    def RetrieveEvaluatedInsertStatesByType(self, action_space, num_features, feature_type):
        if self.lib.IsLeaf(self.node_ptr):
            return None
        state_length = num_features * action_space         # action_space is topk, action_spcae_size
        state_c = (c_double * state_length)()
        self.lib.RetrieveEvaluatedInsertStatesByType(self.tree_ptr, self.node_ptr, self.rec_ptr, action_space, state_c, num_features, feature_type)
        states = np.ctypeslib.as_array(state_c)
        return states
    
    def RetrieveInsertStatesByTypeMCTSVersion(self, node_ptr, rec_ptr, action_space, num_features, feature_type):
        # actually given a node-id, calculate if insert the obj to the node_id
        if self.lib.IsLeaf(node_ptr):
            return None
        state_length = num_features * action_space         # action_space is topk, action_spcae_size
        state_c = (c_double * state_length)()
        self.lib.RetrieveInsertStatesByTypeMCTSVersion(self.tree_ptr, node_ptr, rec_ptr, action_space, state_c, num_features, feature_type)
        states = np.ctypeslib.as_array(state_c)
        return states
    
    def ActionToNodePtr(self, node, action):
        return self.lib.ActionToNodePtr(self.tree_ptr, node, action)

    def GetNumNodes(self):
        """获取树中节点总数"""
        return self.lib.GetNumNodes(self.tree_ptr)
    
    def GetNumObjects(self):
        """获取树中对象（矩形）总数"""
        return self.lib.GetNumObjects(self.tree_ptr)
    
    def GetTotalEntries(self):
        """获取所有节点的 entry 总数"""
        return self.lib.GetTotalEntries(self.tree_ptr)


def tree_variants_query_test(data_size='H1', distribution_type='UNIFORM', query_type='range', num_queries=1000):
    """
    测试不同 RTree 变体在各种数据分布下的查询性能
    
    Parameters
    ----------
    data_size : str
        数据规模标识 (如 'W1', 'H1', 'TW1' 等)
    distribution_type : str
        分布类型 (如 'UNIFORM', 'NORMAL', 'BIMODAL', 'CORRELATED' 等)
    query_type : str
        查询类型：'range' (范围查询) 或 'knn' (KNN 查询)
    num_queries : int
        查询次数
    """
    
    def load_data_from_npy(data_file):
        """从 npy 文件加载数据"""
        try:
            rectangles = np.load(data_file)
            model_dataset = rectangles.tolist()
            print(f"✓ Loaded {len(model_dataset)} rectangles from NPY file: {data_file}")
            return model_dataset
        except Exception as e:
            print(f"✗ Failed to load NPY file: {e}")
            return None
    
    def load_data_from_txt(data_file):
        """从 txt 文件加载数据"""
        model_dataset = []
        try:
            with open(data_file) as input_file:
                n = 0
                ll_x = ll_y = tr_x = tr_y = 0
                for line in input_file:
                    if n % 2 == 0:
                        ll_x = float(line.strip()) - 0.0001
                        tr_x = float(line.strip())
                    else:
                        ll_y = float(line.strip()) - 0.0001
                        tr_y = float(line.strip()) 
                        model_dataset.append([ll_x, ll_y, tr_x, tr_y])
                    n += 1
            print(f"✓ Loaded {len(model_dataset)} rectangles from TXT file: {data_file}")
            return model_dataset
        except Exception as e:
            print(f"✗ Failed to load TXT file: {e}")
            return None
    
    def generate_random_data(num_rects=1000):
        """生成随机测试数据"""
        import random
        model_dataset = []
        for i in range(num_rects):
            base_x = random.uniform(x_min, x_max - 100)
            base_y = random.uniform(y_min, y_max - 100)
            width = random.uniform(50, 200)
            height = random.uniform(50, 200)
            model_dataset.append([base_x, base_y, base_x + width, base_y + height])
        print(f"✓ Generated {len(model_dataset)} random rectangles")
        return model_dataset
    
    def single_tree_query(tree, query_ratio=2.0, knnk=1, query_type='range'):
        """单次查询性能测试"""
        tree_acc_no = 0
        # query_area：百分比 of the whole range
        testing_query_area = query_ratio / 100 * ((x_max - x_min) * (y_max - y_min))    
        side = (testing_query_area**0.5) / 2
        k = 0
        
        while k < num_queries:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            
            if query_type == 'range':
                if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:
                    tree_access = tree.Query((x - side, y - side, x + side, y + side))         
                    tree_acc_no += tree_access
                    k += 1
            elif query_type == 'knn':
                tree_access = tree.KNNQuery(x, y, knnk)
                tree_acc_no += tree_access
                k += 1
                
        return tree_acc_no / num_queries

    # Settings 
    max_entry = 50
    min_entry_factor = 0.4
    # insert_stgy = ["INS_AREA", "INS_RSTAR"]
    insert_stgy = ["INS_AREA", "INS_OVERLAP", "INS_RSTAR"]
    # split_stgy = ["SPL_QUADRATIC", "SPL_MIN_OVERLAP"]
    split_stgy = ["SPL_QUADRATIC", "SPL_QUADRATIC"]
    
    # 查询参数配置
    if query_type == 'range':
        QUERY_PARAMS = [2.0, 1.0, 0.5, 0.05, 0.01, 0.005]  # 查询区域百分比
    else:  # knn
        QUERY_PARAMS = [1, 5, 10, 50, 100, 250, 425]  # K 值
    
    x_min = y_min = 0
    x_max = y_max = 100000

    # Reading data - 优先级：NPY > TXT > Random
    model_dataset = []
    prefix = "testing_" 
    
    # 构建数据文件路径（优先从 test_distributions 目录查找）
    test_dist_dir = os.path.join(generated_data_DIR, "test_distributions")
    data_file_npy = os.path.join(test_dist_dir, f"{prefix}{data_size}_{distribution_type}.npy")
    data_file_txt = os.path.join(test_dist_dir, f"{prefix}{data_size}_{distribution_type}.txt")
    fallback_file_txt = os.path.join(generated_data_DIR, f"{prefix}{data_size}_{distribution_type}.txt")
    
    print("=" * 80)
    print(f"RTree Performance Test - {data_size} Dataset ({distribution_type})")
    print(f"Query Type: {query_type.upper()} Query")
    print("=" * 80)
    print()
    
    # 尝试加载数据（优先级：NPY > TXT > Random）
    print("Loading data...")
    print(f"  Searching in: {test_dist_dir}")
    print(f"  NPY file: {data_file_npy}")
    print(f"  TXT file: {data_file_txt}")
    print(f"  Fallback: {fallback_file_txt}")
    print()
    
    # 列出目录内容以便调试
    if os.path.exists(test_dist_dir):
        files_in_dir = os.listdir(test_dist_dir)
        print(f"  Files in test_distributions/: {len(files_in_dir)} files")
        
        # 查找匹配的候选文件
        matching_files = [f for f in files_in_dir if distribution_type in f]
        if matching_files:
            print(f"  ✓ Found {len(matching_files)} file(s) for '{distribution_type}':")
            for f in matching_files[:5]:  # 显示前 5 个
                print(f"      - {f}")
            if len(matching_files) > 5:
                print(f"      ... and {len(matching_files) - 5} more")
        else:
            print(f"  ⚠️  No files found for distribution '{distribution_type}'")
            
        if len(files_in_dir) <= 20:
            print(f"  All files:")
            for f in files_in_dir:
                print(f"    - {f}")
        else:
            print(f"  (showing first 10 files)")
            for f in files_in_dir[:10]:
                print(f"    - {f}")
    else:
        print(f"  ⚠️  Directory {test_dist_dir} does not exist!")
    print()
    
    # 1. 优先尝试加载 NPY 文件
    if os.path.exists(data_file_npy):
        print(f"✓ Found NPY file: {data_file_npy}")
        model_dataset = load_data_from_npy(data_file_npy)
    
    # 2. 如果 NPY 不存在，尝试加载 TXT 文件（test_distributions 目录）
    if not model_dataset and os.path.exists(data_file_txt):
        print(f"✓ Found TXT file: {data_file_txt}")
        model_dataset = load_data_from_txt(data_file_txt)
    
    # 3. 如果还没有，尝试加载 fallback 的 TXT 文件
    if not model_dataset and os.path.exists(fallback_file_txt):
        print(f"✓ Found fallback TXT file: {fallback_file_txt}")
        model_dataset = load_data_from_txt(fallback_file_txt)
    
    # 4. 如果所有文件都不存在，生成随机数据
    if not model_dataset:
        print(f"⚠️  Warning: No data files found for {data_size} {distribution_type}")
        print(f"   Generating random test data...")
        model_dataset = generate_random_data(1000)
    
    print()
    
    # Create trees
    print("Building RTree variants...")
    traditional_rtree = RTree(max_entry, min_entry_factor)
    traditional_rtree.SetDefaultInsertStrategy(insert_stgy[0])
    traditional_rtree.SetDefaultSplitStrategy(split_stgy[0])

    rstar_tree = RTree(max_entry, min_entry_factor)
    rstar_tree.SetDefaultInsertStrategy(insert_stgy[1])
    rstar_tree.SetDefaultSplitStrategy(split_stgy[1])

    rrstar_tree = RTree(max_entry, min_entry_factor)
    rrstar_tree.SetDefaultInsertStrategy(insert_stgy[1])
    rrstar_tree.SetDefaultSplitStrategy(split_stgy[1])
    
    print(f"✓ Initial tree heights (empty trees):")
    print(f"  Traditional R-Tree: Height={traditional_rtree.GetTreeHeight()}")
    print(f"  R*-Tree: Height={rstar_tree.GetTreeHeight()}")
    print(f"  RR*-Tree: Height={rrstar_tree.GetTreeHeight()}")
    print()

    # Insert data
    print(f"Inserting {len(model_dataset)} rectangles...")
    insert_count = 0
    for insert_obj in model_dataset:
        traditional_rtree.DefaultInsert(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])
        
        rstar_tree.DirectInsert(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])
        rstar_tree.DirectSplitWithReinsert()
        
        rrstar_tree.DirectRRInsert(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])
        rrstar_tree.DirectRRSplit()
        
        insert_count += 1
        if insert_count % 1000 == 0:
            print(f"  Inserted {insert_count}/{len(model_dataset)} rectangles...")
    
    print(f"✓ All trees built successfully")
    print()
    
    # 打印插入后的树高
    print(f"✓ Final tree heights (after inserting {len(model_dataset)} rectangles):")
    print(f"  Traditional R-Tree: Height={traditional_rtree.GetTreeHeight()}")
    print(f"  R*-Tree: Height={rstar_tree.GetTreeHeight()}")
    print(f"  RR*-Tree: Height={rrstar_tree.GetTreeHeight()}")
    print()
    
    # Query
    print(f"Running {query_type.upper()} queries...")
    traditional_results = []
    rstar_results = []
    rrstar_results = []
    
    param_labels = []
    for param in QUERY_PARAMS:
        if query_type == 'range':
            param_labels.append(f"{param}%")
        else:
            param_labels.append(f"k={param}")
        
        traditional_results.append(single_tree_query(traditional_rtree, param if query_type == 'range' else 0, 
                                                      param if query_type == 'knn' else 1, query_type))
        rstar_results.append(single_tree_query(rstar_tree, param if query_type == 'range' else 0, 
                                                param if query_type == 'knn' else 1, query_type))
        rrstar_results.append(single_tree_query(rrstar_tree, param if query_type == 'range' else 0, 
                                                 param if query_type == 'knn' else 1, query_type))
    
    # Print results
    print()
    print("=" * 80)
    print(f"Results - {data_size} {distribution_type} ({query_type.upper()} Query)")
    print("=" * 80)
    print(f"Parameters: {param_labels}")
    print()
    print(f"Traditional R-Tree: {traditional_results}")
    print(f"R*-Tree:           {rstar_results}")
    print(f"RR*-Tree:          {rrstar_results}")
    print()
    
    # Calculate improvements
    if len(traditional_results) > 0:
        avg_trad = sum(traditional_results) / len(traditional_results)
        avg_rstar = sum(rstar_results) / len(rstar_results)
        avg_rrstar = sum(rrstar_results) / len(rrstar_results)
        
        improvement_rstar = ((avg_trad - avg_rstar) / avg_trad) * 100 if avg_trad > 0 else 0
        improvement_rrstar = ((avg_trad - avg_rrstar) / avg_trad) * 100 if avg_trad > 0 else 0
        
        print("-" * 80)
        print(f"Average Node Accesses:")
        print(f"  Traditional R-Tree: {avg_trad:.2f}")
        print(f"  R*-Tree:            {avg_rstar:.2f} (Improvement: {improvement_rstar:+.1f}%)")
        print(f"  RR*-Tree:           {avg_rrstar:.2f} (Improvement: {improvement_rrstar:+.1f}%)")
        print("=" * 80)
    
    print("\n✓ Test completed!")
    return {
        'traditional': traditional_results,
        'rstar': rstar_results,
        'rrstar': rrstar_results,
        'params': param_labels
    }


def comprehensive_distribution_test():
    """
    综合测试：在所有分布类型上测试 RTree 性能
    
    测试所有 15 种分布类型：
    - 基础分布：UNIFORM, NORMAL, SKEW-NOR0/2/4/8
    - 多峰分布：BIMODAL, MULTI-MODAL, CLUSTERED
    - 复杂分布：CORRELATED, ANTI-CORR, GAUSSIAN-MIX
    - 时空数据：SPATIAL-TEMPORAL, CITY-LIKE, HOTSPOTS
    """
    print("=" * 80)
    print("Comprehensive RTree Performance Test - All Distribution Types")
    print("=" * 80)
    print()
    
    # 定义所有要测试的分布类型
    distributions = {
        'Basic Distributions': ['UNIFORM', 'NORMAL', 'SKEW-NOR2'],
        'Multi-modal Distributions': ['BIMODAL', 'MULTI-MODAL', 'CLUSTERED'],
        'Complex Distributions': ['CORRELATED', 'ANTI-CORR', 'GAUSSIAN-MIX'],
        'Spatial-Temporal Data': ['SPATIAL-TEMPORAL', 'CITY-LIKE', 'HOTSPOTS']
    }
    
    # 使用 H1 规模进行快速测试
    data_size = 'W1'
    results = {}
    
    for category, dist_list in distributions.items():
        print(f"\n{'='*80}")
        print(f"Testing Category: {category}")
        print(f"{'='*80}")
        
        results[category] = {}
        
        for dist_type in dist_list:
            print(f"\n>>> Testing {dist_type}...")
            try:
                result = tree_variants_query_test(
                    data_size=data_size,
                    distribution_type=dist_type,
                    query_type='range',
                    num_queries=500  # 减少查询次数以加快测试
                )
                results[category][dist_type] = result
                print(f"✓ {dist_type} completed")
            except Exception as e:
                print(f"✗ {dist_type} failed: {e}")
                results[category][dist_type] = None
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    total_tests = sum(len(v) for v in results.values())
    successful_tests = sum(1 for cat in results.values() for res in cat.values() if res is not None)
    
    print(f"Total distributions tested: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RTree Performance Testing Tool')
    
    # 测试模式选择
    mode_group = parser.add_argument_group('Test Mode')
    mode_group.add_argument('--mode', type=str, default='single', 
                           choices=['single', 'comprehensive'],
                           help='Test mode: single (single distribution) or comprehensive (all distributions)')
    
    # 单分布测试参数
    single_group = parser.add_argument_group('Single Distribution Test Parameters')
    single_group.add_argument('--size', type=str, default='H1',
                             help='Data size (e.g., H1, W1, TW1, etc.)')
    single_group.add_argument('--dist', type=str, default='UNIFORM',
                             help='Distribution type (e.g., UNIFORM, BIMODAL, CORRELATED, etc.)')
    single_group.add_argument('--query', type=str, default='range',
                             choices=['range', 'knn'],
                             help='Query type: range or knn')
    single_group.add_argument('--num-queries', type=int, default=1000,
                             help='Number of queries to perform')
    
    args = parser.parse_args()
    
    if args.mode == 'comprehensive':
        # 综合测试所有分布
        comprehensive_distribution_test()
    else:
        # 单个分布测试
        tree_variants_query_test(
            data_size=args.size,
            distribution_type=args.dist,
            query_type=args.query,
            num_queries=args.num_queries
        )
