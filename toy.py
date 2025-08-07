# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: Optional[ListNode]
        :type l2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        init = ListNode(0)
        prev = ListNode(0)
        plus_one = 0
        

        while l1.next is not None or l2.next is not None:
            a = 0
            if l1.next is not None:
                a += l2.val
            if l2.next is None:
                a += l1.val
            plus_one = a + plus_one // 10
            number = ListNode((a+plus_one)%10)
            if init.next is None:
                init.next = number
            else:
                prev.next = number
            prev = number
            l1 = l1.next
            l2 = l2.next
        if plus_one > 0:
            number.next = ListNode(1)
        if init.next is None:
            return init
        return init.next