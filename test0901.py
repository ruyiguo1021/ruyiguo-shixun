import unittest



def get_nested_depth(lst):
    """
    计算传入的列表（可能包含嵌套列表）的最大嵌套深度。
    例如：[1, 2, 3] 深度为 1；[[1], [2, [3]]] 深度为 3。
    """
    # 如果当前元素本身不是列表，则深度为 0
    if not isinstance(lst, list):
        return 0

    # 如果列表为空，其嵌套深度为 1
    if not lst:
        return 1

    # 递归计算每一个子元素的嵌套深度，取最大值并加上当前这一层（+1）
    return 1 + max(get_nested_depth(item) for item in lst)


# 编写至少5个单元测试用例
class TestNestedDepth(unittest.TestCase):

    def test_depth_1(self):
        """测试用例 1：普通一维列表"""
        self.assertEqual(get_nested_depth([1, 2, 3]), 1)

    def test_depth_2(self):
        """测试用例 2：二维嵌套列表"""
        self.assertEqual(get_nested_depth([1, [2, 3], 4]), 2)

    def test_depth_3(self):
        """测试用例 3：三维嵌套列表（题目示例）"""
        self.assertEqual(get_nested_depth([[1], [2, [3]]]), 3)

    def test_empty_list(self):
        """测试用例 4：空列表"""
        self.assertEqual(get_nested_depth([]), 1)

    def test_deep_nested(self):
        """测试用例 5：多层深层嵌套列表"""
        self.assertEqual(get_nested_depth([1, [2, [3, [4, [5]]]]]), 5)



class User:
    """定义一个用户类"""

    def __init__(self, name):
        self.name = name

    def describe_user(self):
        """打印用户的基本信息"""
        print(f"用户姓名: {self.name}")

    def greet_user(self):
        """打印个性化问候语"""
        print(f"你好，{self.name}！欢迎回来！")


class Admin(User):
    """管理员类，继承自 User 类"""

    def __init__(self, name):
        super().__init__(name)
        # 定义存储权限的列表属性
        self.privileges = ["can add post", "can delete post", "can ban user"]

    def show_privileges(self):
        """显示管理员的权限"""
        print(f"管理员 {self.name} 的权限如下：")
        for privilege in self.privileges:
            print(f"- {privilege}")


if __name__ == '__main__':

    print("--- 正在执行列表深度的单元测试 ---")
    # 运行 unittest 单元测试
    # exit=False 防止测试完成后直接退出程序
    unittest.main(exit=False)

    print("\n--- 正在测试 User 和 Admin 类 ---")
    # 创建 Admin 的实例
    admin_user = Admin("张三")
    import cv2

    print("OpenCV 版本:", cv2.__version__)

    # 调用它的所有方法
    admin_user.describe_user()  # 继承自 User 的方法
    admin_user.greet_user()  # 继承自 User 的方法
    admin_user.show_privileges()  # Admin 自己的方法


