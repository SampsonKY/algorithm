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

