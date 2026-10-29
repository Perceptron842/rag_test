"""LeetCode 11: 盛最多水的容器。双指针 O(n) 时间，O(1) 空间。"""

def max_area(height: list[int]) -> int:
    left,right=0,len(height)-1
    best=0
    while left < right:
        best=max(best, min(height[left],height[right])*(right-left))
        if height[left]<height[right]:
            left+=1
        else:
            right-=1
    return best


if __name__=='__main__':
    assert max_area([1,8,6,2,5,4,8,3,7])==49
    assert max_area([1,1])==1
    assert max_area([])==0
    print('LeetCode 11 示例通过')
