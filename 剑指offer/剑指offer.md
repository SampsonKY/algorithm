## 栈

[剑指 Offer 30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)

**思路**：借助辅助栈

```js
var MinStack = function() {
    this.stack = []
    this.minstack = []
};

MinStack.prototype.push = function(x) {
    this.stack.push(x)
    if(this.minstack.length === 0 || x<this.minstack[this.minstack.length-1]){
        this.minstack.push(x)
    } else{
        this.minstack.push(this.minstack[this.minstack.length-1])
    }
};

MinStack.prototype.pop = function() {
    this.stack.pop()
    this.minstack.pop()
};

MinStack.prototype.top = function() {
    return this.stack[this.stack.length-1]
};

MinStack.prototype.min = function() {
    return this.minstack[this.minstack.length-1]
};
```



## 数组

[剑指 Offer 40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

**题目**：输入整数数组 `arr` ，找出其中最小的 `k` 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

**思路**

- 直接排序
- 利用快排思想
- 大根堆 or 红黑树
- 计数排序

具体参考：[题解](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/solution/chao-quan-3chong-jie-fa-zhi-jie-pai-xu-zui-da-dui-/)

[面试题 01.07. 旋转矩阵](https://leetcode-cn.com/problems/rotate-matrix-lcci/)

**题目**：给你一幅由 `N × N` 矩阵表示的图像，其中每个像素的大小为 4 字节。请你设计一种算法，将图像旋转 90 度。

**示例**：

```
给定 matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

原地旋转输入矩阵，使其变为:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
```

**思路**：

- 方法一：找出翻转前后矩阵的对应关系，利用一个辅助矩阵
- 方法二：原地翻转，先水平翻转，在沿主对角线翻转

```js
//方法一
var rotate = function(matrix) {
    let n = matrix.length
    let new_matrix = new Array(n)
    for(let i=0; i<n; i++){
        new_matrix[i] = new Array(n)
    }
    for(let i=0; i<n; i++){
        for(let j =0; j<n; j++){
            new_matrix[j][n-i-1] = matrix[i][j]
        }
    }
    
    for(let i=0; i<n; i++){
        for(let j=0; j<n; j++){
            matrix[i][j] = new_matrix[i][j]
        }
    }
};

//方法二 
var rotate = function(matrix){
    let n = matrix.length
    //水平翻转
    for(let i=0; i<n/2; i++){
        for(let j=0; j<n; j++){
            let tmp = matrix[i][j]
            matrix[i][j] = matrix[n-i-1][j]
            matrix[n-i-1][j] = tmp
        }
    }

    //主对角线翻转
    for(let i=0; i<n; i++){
        for(let j=0; j<i; j++){
            let tmp = matrix[i][j]
            matrix[i][j] = matrix[j][i]
            matrix[j][i] = tmp
        }
    }
}
```

[剑指 Offer 29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

**题目**：输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

 **示例**：

```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```

**思路**

- 方法一：每一次都将矩阵第一行弹出，然后再逆时针将矩阵翻转90°，循环执行，直到数组为空

- 方法二：初始化矩阵左右上下四个边界，再“从左到右，从上到下，从右到左，从下到上”四个方向循环，每个方向打印做三件事

  - 根据边界打印，即将元素按顺序添加至列表 `res` 尾部；
  - 边界向内收缩 11 （代表已被打印）；
  - 判断是否打印完毕（边界是否相遇），若打印完毕则跳出

  | 打印方向 | 1. 根据边界打印        | 2. 边界向内收缩  | 3. 是否打印完毕 |
  | -------- | ---------------------- | ---------------- | --------------- |
  | 从左向右 | 左边界`l` ，右边界 `r` | 上边界 `t` 加 11 | 是否 `t > b`    |
  | 从上向下 | 上边界 `t` ，下边界`b` | 右边界 `r` 减 11 | 是否 `l > r`    |
  | 从右向左 | 右边界 `r` ，左边界`l` | 下边界 `b` 减 11 | 是否 `t > b`    |
  | 从下向上 | 下边界 `b` ，上边界`t` | 左边界 `l` 加 11 | 是否 `l > r`    |

```js
//方法一：矩阵翻转
var spiralOrder = function(matrix) {
    let len = matrix.length
    if(len===0) return []
    let res = []
    while(matrix.length){
        res.push(...matrix.shift()) //每次将矩阵第一行弹出
        matrix = rotate(matrix)
    }
    function rotate(matrix){ //将矩阵逆时针旋转90°
        if(matrix.length==0) return []
        let n = matrix.length
        let m = matrix[0].length
        let new_matrix = new Array(m)
        for(let j=0; j<m; j++){
            new_matrix[j] = new Array(n)
        }
        for(let i=0; i<n; i++){
            for(let j = 0; j<matrix[0].length; j++){
                new_matrix[m-j-1][i] = matrix[i][j] //关键交换步骤
            }
        }
        return new_matrix
    }
    return res
};
//方法二
function spiralOrder(matrix){
    if(matrix.length === 0) return []
    let l = 0, r=matrix[0].length-1, t=0, b = matrix.length-1,x=0
    let res = []
    while(true){
        for(let i=l; i<=r; i++) res[x++] = matrix[t][i]  //left to right
        if(++t > b) break
        for(let i=t; i<=b; i++) res[x++] = matrix[i][r] //top to bottom
        if(l > --r) break
        for(let i=r; i>=l; i--) res[x++] = matrix[b][i] //right to left
        if(t>--b) break
        for(let i=b; i>=t; i--) res[x++] = matrix[i][l] //bottom to top
        if(++l > r) break
    }
    return res
}
```



## 字符串

[剑指 Offer 50. 第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

**题目**：在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

```js
var firstUniqChar = function(s) {
    //暴力
    // for(let i=0; i<s.length; i++){
    //     let tmp = s[i]
    //     let flag = 1
    //     for(var j=0; j<s.length; j++){
    //         if(i!==j && s[j] === s[i]) {
    //             break
    //         }
    //     }
    //     if(j==s.length) return s[i]
    // }
    // return " "
    
    //map
    // let map = new Map()
    // for(let i=0; i<s.length; i++){
    //     map.has(s[i]) ? map.set(s[i], false) : map.set(s[i],true)
    // }
    // for(let [key, value] of map.entries()){
    //     if(value) return key
    // }
    // return " "
    
    //对象
    let obj = {}
    for(let i=0; i<s.length; i++){
        obj.hasOwnProperty(s[i]) ? obj[s[i]] = false : obj[s[i]] = true
    }
    for(let item in obj){
        if(obj[item]) return item
    }
    return " "
};
```

[剑指 Offer 58 - I. 翻转单词顺序](https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

**示例 1：**

```
输入: "  hello world!  "
输出: "world! hello"
解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
```

**示例 2：**

```
输入: "a good   example"
输出: "example good a"
解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
```

**思路：**

- 直接利用字符串的方法 +正则
- 双指针法

```js
//法一
var reverseWords = function(s) {
    return s.trim().replace(/\s+/ig," ").split(" ").reverse().join(" ")
};

//双指针法
var reverseWords = function(s) {
    s = s.trim()
    let left = right = s.length-1
    let res = ""
    while(left >= 0){
        while(left>=0 && s[left]!==" ") left--
        res+=(s.substring(left+1,right+1)+ " ")
        while(left>=0 && s[left]===" ") left--
        right = left
    }
    return res.trim()
};
```



## 链表

[剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

**题目**：输入两个链表，找出它们的第一个公共节点。

```js
var getIntersectionNode = function(headA, headB) {
    let h1 = headA, h2 = headB
    while(h1 !== h2){
        h1 = h1===null? headB: h1.next
        h2 = h2===null?headA: h2.next
    }
    return h2
};
```

**[剑指 Offer 24. 反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)**

```js
var reverseList = function(head) {
    if(!head) return null
    let cur = head, prev = null
    while(cur){
        let next = cur.next
        cur.next = prev
        prev = cur
        cur = next
    }
    return prev
};
```

[剑指 Offer 25. 合并两个排序的链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

```js
var mergeTwoLists = function(l1, l2) {
    let res = p = new ListNode()
    while(l1 || l2){
        if(!l1) {p.next = l2; break}
        if(!l2) {p.next = l1; break}
        if(l1.val < l2.val){
            p.next = l1
            p = p.next
            l1 = l1.next
        } else{
            p.next = l2
            p = p.next
            l2 = l2.next
        }
    }
    return res.next
};
```

[剑指 Offer 35. 复杂链表的复制](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

**题目**：请实现 `copyRandomList` 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 `next` 指针指向下一个节点，还有一个 `random` 指针指向链表中的任意节点或者 `null`。

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e1.png)

```
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
```

**思路**：先复制next，再复制 random

```js
//方法一：借助map
var copyRandomList = function(head) {
    if(!head) return null
    var cur = head
    var map = new Map()
    while(cur){
        var clone = new Node(cur.val, null, null)
        map.set(cur, clone)
        cur = cur.next
    }
    cur = head
    while(cur){
        map.get(cur).next = cur.next===null?null:map.get(cur.next)
        map.get(cur).random = map.get(cur.random)
        cur = cur.next
    }
    return map.get(head)
};

//方法二
var copyRandomList = function(head){
    if(!head) return null
    var cur = head
    while(cur){
        var clone = new Node(cur.val, cur.next, null)
        var tmp = cur.next
        cur.next = clone
        cur = tmp
    }
    cur = head
    while(cur){
        if(cur.random) cur.next.random = cur.random.next
        cur = cur.next.next
    }
    cur = head
    var newlist = cur.next
    while(cur && cur.next){
        var tmp = cur.next
        cur.next = cur.next.next
        cur = tmp
    }
    return newlist
}
```



### 快慢指针

[剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

```js
var getKthFromEnd = function(head, k) {
    let slow = fast = head
    for(let i=0; i<k; i++){
        fast = fast.next
    }
    while(fast){
        fast = fast.next
        slow = slow.next
    }
    return slow
};
```



## 二叉树

[剑指 Offer 68 - II. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/15/binarytree.png)

**思路**

若两个指定节点分别在树的两边，则根节点即为最近公共祖先；若均在左子树，则递归遍历左子树；均在右子树，则递归右子树

```js
function lowestCommonAncestor(root, p, q){
    if(!root) return null
    if(find(root.left, p) && find(root.left, q)) root = lowestCommonAncestor(root.left, p, q)
    if(find(root.right, p) && find(root.right, q)) root = lowestCommonAncestor(root.right, p, q)
    return root
};

function find(root, node){
    if(!root) return false
    if(root.val === node) return true
    return find(root.left, node) || find(root.right, node)
}
```

[剑指 Offer 55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)

**思路**

- 方法1：递归
- 方法2： BFS

```js
//递归
var maxDepth = function(root) {
    if(!root) return 0
    let left = maxDepth(root.left)
    let right = maxDepth(root.right)
    return Math.max(right, left) + 1
};
//BFS
var maxDepth = function(root){
    if(!root) return 0
    let queue = []
    let step = 0
    queue.push(root)

    while(queue.length){
        let size = queue.length
        while(size--){
            let cur = queue.shift()
            if(cur.left) queue.push(cur.left)
            if(cur.right) queue.push(cur.right)
        }
        step++
    }
    return step
}
```

[剑指 Offer 55 - II. 平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/)

```js
var isBalanced = function(root) {
    if(!root) return true
    let left = Depth(root.left)
    let right = Depth(root.right)
    if(Math.abs(left-right)>1) return false
    return isBalanced(root.left) && isBalanced(root.right)
};

var Depth = function(root){
    if(!root) return 0
    var left = Depth(root.left)
    var right = Depth(root.right)
    var depth = Math.max(left, right)+1
    return depth
}
```

[剑指 Offer 28. 对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)

```js
var isSymmetric = function(root) {
    if(!root) return true
    return compare(root.left, root.right)
};
var compare = function(s, t){
    if(!s && !t) return true
    if(!s || !t) return false
    if(s.val !== t.val) return false
    return compare(s.left,t.right) && compare(s.right, t.left)
}
```





### 二叉搜索树

[剑指 Offer 68 - I. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

```js
var lowestCommonAncestor = function(root, p, q) {
    if(!root) return null
    if(root.val >p.val && root.val >q.val) root = lowestCommonAncestor(root.left, p,q)
    if(root.val <p.val && root.val <q.val) root = lowestCommonAncestor(root.right, p,q)
    return root
};
```



## 双指针

[剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

**题目**：输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

```js
var exchange = function(nums) {
    let left = 0, right = nums.length-1
    while(left < right){
        if(nums[left] %2 !== 0){
            left++
        }
        if(nums[right]%2 ==0){
            right--
        }
        if(left < right){
            let tmp = nums[left]
            nums[left] = nums[right]
            nums[right] = tmp
        }
    }
    return nums
};
```

[剑指 Offer 57. 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

**题目**：输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

```js
var twoSum = function(nums, target) {
    let i = 0, j= nums.length-1
    while(i<j){
        let sum = nums[i]+nums[j]
        if(sum === target){
            return [nums[i],nums[j]]
        } else if(sum > target){
            j--
        } else{
            i++
        }
    }
    return []
};
```

这题也可以用哈希表法。



## 滑动窗口

[剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

**题目**：输入一个正整数 `target` ，输出所有和为 `target` 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

**示例 1：**

```
输入：target = 9
输出：[[2,3,4],[4,5]]
```

```js
var findContinuousSequence = function(target) {
    let res = []
    for(let l=1,r=2; l<r;){
        let sum = (l+r)*(r-l+1)/2
        if(sum === target){
            let tmparr = []
            for(let i=l; i<=r; i++){
                tmparr.push(i)
            }
            res.push(tmparr)
            l++
        }else if(sum<target){
            r++
        }else{
            l++
        }
    }
    return res
};
```



## 二进制

[剑指 Offer 15. 二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

请实现一个函数，输入一个整数，输出该数二进制表示中 1 的个数。例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。

**示例 1：**

```
输入：00000000000000000000000000001011
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
```

**思路**

* 方法一：**逐位判断**
  - 根据与运算定义，设二进制数字n，则有：
    - 若 n \& 1 = 0 ，则 n二进制 **最右一位** 为 0 ；
    - 若 n \& 1 = 1 ，则 n*二进制 **最右一位** 为 1 。
  - 根据以上特点，考虑以循环判断
    1. 判断 *n* 最右一位是否为 1 ，根据结果计数。
    2. 将 *n* 右移一位
* 方法二：**巧用`n&(n-1)`**
  - (n−1) 解析： 二进制数字 n 最右边的 1 变成 00 ，此 1 右边的 0 都变成 1 。
  - n&(n−1) 解析： 二进制数字 n 最右边的 1 变成 0 ，其余不变。

```js
//方法1
var hammingWeight = function(n) {
    let res = 0
    while(n!=0){
        res += n&1
        n>>>=1
    }
    return res
};
//方法2
var hammingWeight = function(n) {
    let res = 0
    while(n!=0){
        res ++
        n &= n-1
    }
    return res
};
```

[剑指 Offer 65. 不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

```js
var add = function(a, b) {
    let sum, carry
    do{
        sum = a^b //相加但不产生进位
        carry = (a&b)<<1 //记下进位
        //前两个步骤相加
        a=sum
        b=carry
    }while(b!=0)

    return a
};
```

[剑指 Offer 56 - II. 数组中数字出现的次数 II](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/)

**题目**：在一个数组 `nums` 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

**示例 ：**

```
输入：nums = [3,4,3,3]
输出：4
```

**思路**：

- 使用 **与运算** ，可以获取二进制数字 num 的最右一位
- 配合 **无符号右移操作**，可以获取 num 所有位的值
- 建立一个长度为 32 的数组 counts，通过上面方法记录所有数字的各二进制位的 1 出现的次数。
- 将 counts 各元素对 3 求余，则结果为“只出现一次的数字”的各二进制位。
- 利用 **左移操作** 和 **或运算** ，可以将 counts 数组中各二进制位的值恢复到数字 res 上
- 最终返回res

```js
var singleNumber = function(nums) {
    let counts = new Array(32).fill(0)
    for(let i=0; i<nums.length; i++){
        for(let j=0; j<32; j++){
            counts[j] += nums[i] & 1
            nums[i] >>>=1
        }
    }

    let res =0
    for(let i=0; i<32; i++){
        res <<=1
        res |= counts[31-i]%3
    }
    return res
};
```



## BFS

[剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

```js
var levelOrder = function(root) {
    if(!root) return []
    let res = []
    let queue = []
    queue.push(root)

    while(queue.length){
        let size = queue.length
        let tmp = []
        while(size--){
            let cur = queue.shift()
            tmp.push(cur.val)
            if(cur.left) queue.push(cur.left)
            if(cur.right) queue.push(cur.right)
        }
        res.push(tmp)
    }
    return res
};
```



## 二分查找

[剑指 Offer 11. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

**题目**：把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 `[3,4,5,1,2]` 为 `[1,2,3,4,5]` 的一个旋转，该数组的最小值为1。 

```js
var minArray = function(numbers) {
    let left = 0, right = numbers.length-1

    while(left <= right){
        let mid = Math.floor((right+left)/2)
        if(numbers[mid] > numbers[right]){
            left = mid+1
        }else if(numbers[mid] < numbers[right]){
            right = mid
        } else right = right-1
    }
    
    return numbers[left]
};
```

[剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

**题目**：统计一个数字在排序数组中出现的次数。

**思路**：找到左边界

 ```js
var search = function(nums, target) {
    let left = 0, right = nums.length-1
    let res = 0
    while(left<=right){
        let mid = Math.floor((left+right)/2)
        if(nums[mid] === target) {
            right = mid-1
        }
        else if(nums[mid] > target){
            right = mid-1
        }else if(nums[mid] < target){
            left = mid+1
        }
    }
    while(nums[left]==target){
        res++
        left++
    }
    return res
};
 ```

[剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

**题目**：一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字

**示例 1:**

```
输入: [0,1,3]
输出: 2
```

```js
var missingNumber = function(nums) {
    let left = 0, right = nums.length-1
    while(left<right){
        let mid = Math.floor((right+left)/2)
        if(nums[mid] !== mid) right = mid
        else left = mid+1
    }
    return left === nums.length-1 && nums[left]==left ? left+1 :left
};
```





## 数学

**[剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)**

**题目**：【约瑟夫环】

0,1,,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

**思路**

<img src="https://pic.leetcode-cn.com/d7768194055df1c3d3f6b503468704606134231de62b4ea4b9bdeda7c58232f4-%E7%BA%A6%E7%91%9F%E5%A4%AB%E7%8E%AF1.png" alt="约瑟夫环1.png" style="zoom:67%;" />

<img src="https://pic.leetcode-cn.com/68509352d82d4a19678ed67a5bde338f86c7d0da730e3a69546f6fa61fb0063c-%E7%BA%A6%E7%91%9F%E5%A4%AB%E7%8E%AF2.png" alt="约瑟夫环2.png" style="zoom:67%;" />使用逆推法：

```js
var lastRemaining = function(n, m) {
    if(n<=0) return -1
    let pos = 0
    for(let i=2; i<=n; i++){
        pos = [pos+m]%i
    }
    return pos
};
```

暴力法（会超时）：

```js
var lastRemaining = function(n, m) {
    if(n<=0) return -1
    const remove = (arr, index)=>{
        for(let i=index; i<arr.length-1; i++){
            arr[i] = arr[i+1]
        }
        return arr
    }
    let s = 0
    let res = []
    for(let i=0; i<n; i++){
        res.push(i)
    }
    let idx = 0
    while(n>1){
        idx = (idx + m-1)%n
        res = remove(res, idx)
        n--
    }

    return res[0]
};
```



**[剑指 Offer 39. 数组中出现次数超过一半的数字](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)**

**题目**：数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

**思路**：【摩尔投票法】，本题也可使用哈希表法

> 设输入数组 `nums` 的众数为 *x* ，数组长度为 n 。

**推论一：** 若记 **众数** 的票数为 +1 ，**非众数** 的票数为 −1 ，则一定有所有数字的 **票数和 >0** 。

**推论二：** 若数组的前 *a* 个数字的 **票数和 =0** ，则 数组剩余 (*n*−*a*) 个数字的 **票数和一定仍 >0** ，即后 (*n*−*a*) 个数字的 **众数仍为 x** 。

1. **初始化：** 票数统计 `votes = 0` ， 众数 `x`；
2. 循环：遍历数组`nums`中的每个数字`num`；
   1. 当 票数 `votes` 等于 0 ，则假设当前数字 `num` 是众数；
   2. 当 `num = x` 时，票数 `votes` 自增 1 ；当 `num != x` 时，票数 `votes` 自减 1 ；
3. **返回值：** 返回 `x` 即可；

```js
var majorityElement = function(nums) {
    // 对象保存
    // let obj = {}
    // for(let i=0; i<nums.length; i++){
    //     obj.hasOwnProperty(nums[i])? obj[nums[i]]++:obj[nums[i]]=1
    //     if(obj[nums[i]] > nums.length/2) return nums[i]
    // }
    // 摩尔投票法
    let res = 0, count = 0
    for(let i=0; i<nums.length; i++){
        if(count==0) res = nums[i]
        res == nums[i] ? count++ : count--
    }
    return res
};
```



## 技巧

**[剑指 Offer 03. 数组中重复的数字](https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)**

**技巧**：【原地置换】，这题也可以用哈希表法

**题目**：在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

**思路**：

遍历数组 nums ，设索引初始值为 i = 0 :

- 若 nums[i] = i： 说明此数字已在对应索引位置，无需交换，因此跳过；
- 若 nums[nums[i]] = nums[i] ： 代表索引 nums[i] 处和索引 i 处的元素值都为 nums[i] ，即找到一组重复值，返回此值 nums[i] ；
- 否则： 交换索引为 i 和 nums[i] 的元素值，将此数字交换至对应索引位置。

若遍历完毕尚未返回，则返回 −1 。

```js
// 如果没有重复数字，那么正常排序后，数字i应该在下标为i的位置，所以思路是重头扫描数组，遇到下标为i的数字如果不是i的话，（假设为m),那么我们就拿与下标m的数字交换。在交换过程中，如果有重复的数字发生，那么终止返回ture
var findRepeatNumber = function(nums) {
    let tmp 
    for(let i=0; i<nums.length; i++){
        while(nums[i]!==i){
            if(nums[i] === nums[nums[i]]){
                return nums[i]
            }
            tmp = nums[i]
            nums[i] = nums[tmp]
            nums[tmp] = tmp
        }
    }
    return -1
};
```

[剑指 Offer 61. 扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

**题目**：从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

**示例**

```
输入: [1,2,3,4,5]
输出: True

输入: [0,0,1,2,5]
输出: True
```

**思路**：5张牌是顺子的充分条件是：① 除大小王外，所有牌**无重复**；②设5张牌中最大的牌为max，最小的牌为min（大小王除外），需要满足**max-min<5**。

```js
//方法一： set + 遍历
var isStraight = function(nums) {
    let set = new Set()
    let max = 0, min = 14
    for(let i=0; i<nums.length; i++){
        if(nums[i] == 0) continue //跳过大小王
        max = Math.max(max, nums[i]) //最大牌
        min = Math.min(min, nums[i]) //最小牌
        if(set.has(nums[i])) return false //若有重复，提前返回flase
        set.add(nums[i])
    }
    return max-min < 5
};

//方法二：排序 + 遍历
var isStraight = function(nums){
    nums.sort((a,b)=>a-b)
    let min = 0
    for(let i=0; i<nums.length; i++){
        if(nums[i]===0) min++
        else if(nums[i+1] === nums[i]) return false
    }
    return nums[4] - nums[min] < 5
}
```

[剑指 Offer 64. 求1+2+…+n](https://leetcode-cn.com/problems/qiu-12n-lcof/)

**题目**：求 `1+2+...+n` ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

**思路**：巧妙利用逻辑与短路特性作为递归结束条件。

```js
var sumNums = function(n) {
    let sum=n
    n && (sum+=sumNums(n-1))
    return sum
};
```



## 大数处理

**[剑指 Offer 17. 打印从1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)**

**题目**：输入数字 `n`，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

**思路**：

- 方法1： **递归生成全排列**
- 方法2：**借助数组或字符串模拟数字加法**

```js
//方法1
var printNumbers = function(n){
    if(n<=0) return []
    let number = new Array(n).fill(0)
    let res = []

    for(let i=0; i<10; i++){
        number[0] = i
        recur(number, n, 0)
    }
    function recur(number, length, index){
        //递归结束的条件是我们已经设置了数字最后一位
        if(index === length -1) {
            res.push(number.join(''))
            return    
        }
        for(let i=0; i<10; i++){
            number[index+1] = i
            recur(number, length, index+1)
        }
    }
    res.shift() // 将0弹出
    return res
}

//方法2
var printNumbers = function(n) {
    if(n<=0) return []
    let number = new Array(n).fill(0)
    let res = []
    while(!Increment(number)){
        res.push(number.join(""))
    }
    return res
};

function Increment(number){
        let isOverflow = false //判断是否溢出
        let nTakeOver = 0 //进位
        let nLength = number.length //n
        for(let i=nLength-1; i>=0; i--){
            let nSum = number[i] + nTakeOver
            if(i===nLength-1) nSum++
            if(nSum>=10){
                if(i===0) isOverflow = true
                else{
                    nSum -= 10
                    nTakeOver = 1
                    number[i] = nSum
                }
            } else{
                number[i] = nSum
                break
            }
        }
        return isOverflow
    }
```



## 动态规划

[剑指 Offer 42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

**题目**：输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。要求时间复杂度为O(n)。

**示例**：

```
输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

**思路**

设`dp[i]`为以`arr[i]`结尾的连续子数组最大和。

- 当 dp[i - 1] > 0 时：执行 dp[i] = dp[i-1] + nums[i]
- 当 dp[i−1]≤0 时：执行 dp[i] = nums[i]；

```js
var maxSubArray = function(nums) {
    let dp = []
    dp[0] = nums[0]
    for(let i=1; i<nums.length; i++){
        dp[i] = dp[i-1]>0 ? dp[i-1]+nums[i] : nums[i]
    }
    return Math.max(...dp)
};
```

**[剑指 Offer 60. n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)**

**题目**：把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。

**示例**：

```
输入: 1
输出: [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667]
```

**思路**

n 个骰子，共有 6^n^ 种排列。点数之和的范围为：n~6*n，即共有(6n-n+1)种情况。

- **方法一：递归**

  把n个骰子分为两堆，第一堆1个，第二堆n-1个。

- **方法二：动态规划**[参考](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/solution/nge-tou-zi-de-dian-shu-dong-tai-gui-hua-ji-qi-yo-3/)

  - `dp[i][j]` 表示投掷完 i 枚骰子后，点数 j 出现的次数。
  - **状态转移方程**

  ![image-20201101215332707](C:\Users\theon\AppData\Roaming\Typora\typora-user-images\image-20201101215332707.png)

  ​	*n* 表示阶段，j表示投掷完 n枚骰子后的点数和，*i* 表示第 *n* 枚骰子会出现的六个点数。

  - **边界处理、初始化**

    初始化状态为投掷完 1 枚骰子后的次数

```js
/**
 * @param {number} n
 * @return {number[]}
 */
//递归
var twoSum = function(n) {
    if(n<1) return []
    let res = new Array(6*n-n+1).fill(0) //n个骰子的点数和最小为n，最大为6n，共有6n-n+1种点数
    Probability(n, res)
    let total = Math.pow(6, n) //总共有6^n中排列情况
    for(let i=n; i<=6*n; i++){ 
        res[i-n] = res[i-n]/total 
    }
    return res
};

function Probability(n, res){
    for(let i=1; i<=6; i++){  //把n个骰子分为两堆，第一堆1个，第二堆n-1个。单独的一个可能出现1~6的点数。需要计算
        Probable(n, n, i, res)
    }
}
function Probable(origin, cur, sum, res){
    if(cur===1) res[sum-origin]++ //递归结束条件就是最后只剩下一个骰子
    else{
        for(let i=1; i<=6; i++){
            Probable(origin, cur-1, i+sum, res)
        }
    }
}

//动态规划
var twoSum = function(n){
    if(n<1) return []
    let dp = new Array(n+1)
    for(let i=0; i<=n;i++){
        dp[i] = new Array(6*n+1).fill(0) 
    }
    for(let i=1; i<=6; i++){
        dp[1][i] = 1
    }
    for(let i=2; i<=n; i++){
        for(let j=i; j<=6*i; j++){
            for(let cur = 1; cur<=6; cur++){
                if(j-cur<=0) break
                dp[i][j] += dp[i-1][j-cur]
            }
        }
    }
    let total = Math.pow(6, n)
    let res = []
    for(let i=n; i<=6*n; i++){
        console.log(dp[n][i])
        res.push(dp[n][i]/total)
    }
    return res
}

//动态规划空间优化
var twoSum = function(n){
    if(n<1) return []
    let dp = new Array(70).fill(0)
    for(let i=1; i<=6; i++){
        dp[i] = 1
    }

    for(let i=2; i<=n; i++){
        for(let j=6*i; j>=i; j--){
            dp[j] = 0
            for(let cur=1; cur<=6; cur++){
                if(j-cur<i-1) break
                dp[j]+=dp[j-cur]
            }
        }
    }
    let total = Math.pow(6,n)
    let res = []
    for(let i=n; i<=6*n; i++){
        res.push(dp[i]/total)
    }
    return res
}
```

