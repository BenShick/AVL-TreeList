import random
"""A class represnting a node in an AVL tree"""


class AVLNode(object):
    """
    @type value: str
    @param value: data of your node
    """
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1
        self.size = 0


        """returns the left child
    @rtype: AVLNode
    @returns: the left child of self, None if there is no left child
    """

    def getLeft(self):
        return self.left

    """returns the right child
    
    @rtype: AVLNode
    @returns: the right child of self, None if there is no right child
    """

    def getRight(self):
        return self.right

    """returns the parent 

    @rtype: AVLNode
    @returns: the parent of self, None if there is no parent
    """

    def getParent(self):
        return self.parent

    """return the value

    @rtype: str
    @returns: the value of self, None if the node is virtual
    """

    def getValue(self):
        return self.value

    """returns the height

    @rtype: int
    @returns: the height of self, -1 if the node is virtual
    """

    def getHeight(self):
        return self.height

    """returns the size of the node

    @rtype: int
    @returns: the size of self, 0 if the node is virtual
    """

    def getSize(self):
        return self.size

    """sets left child

    @type node: AVLNode
    @param node: a node
    """

    def setLeft(self, node):
        self.left = node

    """sets right child

    @type node: AVLNode
    @param node: a node
    """

    def setRight(self, node):
        self.right = node

    """sets parent

    @type node: AVLNode
    @param node: a node
    """

    def setParent(self, node):
        self.parent = node

    """sets value

    @type value: str
    @param value: data
    """

    def setValue(self, value):
        self.value = value

    """sets the balance factor of the node

    @type h: int
    @param h: the height
    """

    def setHeight(self, h):
        self.height = h

    """sets size

    @type i: int
    @param i: new size of the node
    """

    def setSize(self, i):
        self.size = i

    """returns whether self is not a virtual node 

    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """

    def isRealNode(self):
        if self.getLeft() is None and self.getRight() is None:
            return False
        return True

    """returns the BF of the node

    @rtype = int
    @returns: balance factor of the node
    """

    def BF(self):
        if self.isRealNode() is False:
            return 0
        return self.left.height - self.right.height


"""
A class implementing the ADT list, using an AVL tree.
"""


class AVLTreeList(object):
    """
    Constructor, you are allowed to add more fields.

    """

    def __init__(self):
        self.size = 0
        self.root = None
        self.firstItem = None
        self.lastItem = None

    """returns whether the list is empty

    @rtype: bool
    @returns: True if the list is empty, False otherwise
    """

    def empty(self):
        if self.size == 0:
            return True
        return False

    """retrieves the value of the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: index in the list
    @rtype: str
    @returns: the the value of the i'th item in the list
    """

    def retrieve(self, i):
        if i > self.size - 1 or i < 0:
            return None
        return self.treeSelect(i + 1).getValue()

    """inserts val at position i in the list

    @type i: int
    @pre: 0 <= i <= self.length()
    @param i: The intended index in the list to which we insert val
    @type val: str
    @param val: the value we inserts
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def insert(self, i, val):
        newNode = AVLNode(val)
        virt1 = AVLNode(None)
        virt2 = AVLNode(None)
        self.insertAsRightSon(son=virt1, parent=newNode)
        self.insertAsLeftSon(son=virt2, parent=newNode)
        if self.empty():
            self.insertToEmptyList(newNode)  # self.firstItem and self.lastItem are updated in this function
        else:
            if i == self.size:
                self.insertLast(newNode)  # self.lastItem is updated in this function
            else:
                nextNode = self.treeSelect(i + 1)
                if nextNode.getLeft().isRealNode() is False:
                    self.insertAsLeftSon(son=newNode, parent=nextNode)
                else:
                    nextNode = self.predecessor(nextNode)
                    self.insertAsRightSon(son=newNode, parent=nextNode)
        if i == 0:
            self.firstItem = newNode
        numOfRotations = self.updateAfterOp(newNode)
        self.size += 1
        return numOfRotations

    """insert son as right son of parent
    
    @type parent: AVLNode    	
    @type son: AVLNode
    """

    def insertAsRightSon(self, son, parent):
        parent.setRight(son)
        son.setParent(parent)

    """insert son as left son of parent
    
    @type parent: AVLNode
    @type son: AVLNode
    """

    def insertAsLeftSon(self, son, parent):
        parent.setLeft(son)
        son.setParent(parent)

    """update the sizes of nodes after operation

    @type node: AVLNode
    @param node: the node we start fixing from
    """

    def updateAfterOp(self, node):
        temp = node
        numOfRotations = 0
        while temp is not None:
            nextHeight = max(temp.getLeft().getHeight(), temp.getRight().getHeight()) + 1
            temp.setHeight(nextHeight)
            temp.setSize(temp.getLeft().getSize() + temp.getRight().getSize() + 1)
            left = temp.getLeft()
            leftBF = left.BF()
            right = temp.getRight()
            rightBF = right.BF()
            if temp.BF() == 2:
                if leftBF == 1 or leftBF == 0:
                    self.rightRotation(temp)
                    numOfRotations += 1
                    self.fixingAfterOneRotation(parent=temp, child=left)
                    temp = left
                if leftBF == -1:
                    grandchild = left.getRight()
                    self.leftThenRightRotation(temp)
                    numOfRotations += 2
                    self.fixingAfterDoubleRotation(parent=temp, child=left, grandchild=grandchild)
                    temp = grandchild
            if temp.BF() == -2:
                if rightBF == -1 or rightBF == 0:
                    self.leftRotation(temp)
                    numOfRotations += 1
                    self.fixingAfterOneRotation(parent=temp, child=right)
                    temp = right
                if rightBF == 1:
                    grandchild = right.getLeft()
                    self.rightThenLeftRotation(temp)
                    numOfRotations += 2
                    self.fixingAfterDoubleRotation(parent=temp, child=right, grandchild=grandchild)
                    temp = grandchild
            temp = temp.getParent()
        return numOfRotations

    """fixing size and height fields of AVL criminal and his child after one rotation
    @type parent: AVLNode
    @type child: AVLNode
    @pre: child.getParent() == parent, parent.BF() == |2|, child.BF() == |1| or 0
    """

    def fixingAfterOneRotation(self, parent, child):
        parent.setHeight(max(parent.getLeft().getHeight(), parent.getRight().getHeight()) + 1)
        parent.setSize(parent.getLeft().getSize() + parent.getRight().getSize() + 1)
        child.setHeight(max(child.getLeft().getHeight(), child.getRight().getHeight()) + 1)
        child.setSize(child.getLeft().getSize() + child.getRight().getSize() + 1)

    """fixing size and height fields of AVL criminal and his child and grandchild after Double rotation

    @type parent: AVLNode
    @type child: AVLNode
    @type grandchild: AVLNode
    @pre: child.getParent() == parent, grandchild.getParent() == child, parent.BF() == |2|, child.BF() == |1| or 0
    """

    def fixingAfterDoubleRotation(self, parent, child, grandchild):
        parent.setHeight(max(parent.getLeft().getHeight(), parent.getRight().getHeight()) + 1)
        parent.setSize(parent.getLeft().getSize() + parent.getRight().getSize() + 1)
        child.setHeight(max(child.getLeft().getHeight(), child.getRight().getHeight()) + 1)
        child.setSize(child.getLeft().getSize() + child.getRight().getSize() + 1)
        grandchild.setHeight(max(grandchild.getLeft().getHeight(), grandchild.getRight().getHeight()) + 1)
        grandchild.setSize(grandchild.getLeft().getSize() + grandchild.getRight().getSize() + 1)

    """retrieves the node of the i'th item in the list

    @type i: int
    @type node: AVLNode
    @pre: 0 <= i < self.length()
    @param i: index in the list
    @param node: the node we start from
    @rtype: AVLNode
    @returns: the node of the i'th item in the list
    """

    def treeSelect(self, i):
        return self.treeSelectRec(self.root, i)

    def treeSelectRec(self, node, i):
        LeftTreeSize = node.getLeft().getSize() + 1
        if LeftTreeSize == i:
            return node
        else:
            if i < LeftTreeSize:
                return self.treeSelectRec(node.getLeft(), i)
            else:
                return self.treeSelectRec(node.getRight(), i - LeftTreeSize)

    """retrieves the predecessor of node

    @type node: AVLNode
    @pre: the node is in the tree
    @rtype: AVLNode
    @returns: the predecessor of node
    """

    def predecessor(self, node):
        if node.getLeft().isRealNode() is False:
            temp = node
            while temp is not None:
                currParent = temp.getParent()
                if currParent.getRight() == temp:
                    return currParent
                else:
                    if currParent.getLeft() == temp:
                        temp = currParent
        if node.getLeft().isRealNode():
            temp = node.getLeft()
            while temp.getRight().isRealNode():
                temp = temp.getRight()
            return temp
        return None

    """insert new node to empty list

    @type newNode: AVLNode
    """

    def insertToEmptyList(self, newNode):
        self.root = newNode
        self.firstItem = newNode
        self.lastItem = newNode

    """insert new node as last

    @type newNode: AVLNode
    """

    def insertLast(self, newNode):
        temp = self.lastItem
        self.insertAsRightSon(son=newNode, parent=temp)
        self.lastItem = newNode

    """Delete leaf

    @type node: AVLNode
    @type isRoot: boolean
    @type parent: AVLNode
    @type isLeftChild: boolean
    """

    def deleteLeaf(self, node, isRoot, parent, isLeftChild):
        if not isRoot:
            if isLeftChild:
                virt1 = AVLNode(None)
                self.insertAsLeftSon(son=virt1, parent=parent)
            else:
                virt2 = AVLNode(None)
                self.insertAsRightSon(son=virt2, parent=parent)
        node.setParent(None)
        node.setRight(None)
        node.setLeft(None)

    """Delete node with left child

    @type node: AVLNode
    @type isRoot: boolean
    @type parent: AVLNode
    @type isLeftChild: boolean
    """

    def deleteNodeWithLeftChild(self, node, isRoot, parent, isLeftChild):
        leftChild = node.getLeft()
        if not isRoot:
            if isLeftChild:
                parent.setLeft(leftChild)
            else:
                parent.setRight(leftChild)
        leftChild.setParent(parent)
        if isRoot:
            self.root = leftChild
        node.setParent(None)
        node.setRight(None)
        node.setLeft(None)

    """Delete node with right child

    @type node: AVLNode
    @type isRoot: boolean
    @type parent: AVLNode
    @type isLeftChild: boolean
    """

    def deleteNodeWithRightChild(self, node, isRoot, parent, isLeftChild):
        rightChild = node.getRight()
        if not isRoot:
            if isLeftChild:
                parent.setLeft(rightChild)
            else:
                parent.setRight(rightChild)
        rightChild.setParent(parent)
        if isRoot:
            self.root = rightChild
        node.setParent(None)
        node.setRight(None)
        node.setLeft(None)

    """Delete node with two children

    @type node: AVLNode
    @type isRoot: boolean
    @type parent: AVLNode
    @type isLeftChild: boolean
    """

    def deleteNodeWithTwoChildren(self, node, isRoot, parent, isLeftChild):
        successor = self.successor(node)
        if successor == node.getRight():
            nodeRight = successor.getRight()
        else:
            nodeRight = node.getRight()
        nodeParent = parent
        nodeIsLeftChild = isLeftChild
        nodeLeft = node.getLeft()
        succParent = successor.getParent()
        succIsRoot = False
        if successor == succParent.getLeft():
            succIsLeftChild = True
        else:
            succIsLeftChild = False
        """the successor has only right child"""
        if successor.getRight().isRealNode() is True:
            self.deleteNodeWithRightChild(successor, succIsRoot, succParent, succIsLeftChild)
        else:
            """the successor is a leaf"""
            self.deleteLeaf(successor, succIsRoot, succParent, succIsLeftChild)
        successor.setParent(nodeParent)
        self.insertAsRightSon(son=nodeRight, parent=successor)
        self.insertAsLeftSon(son=nodeLeft, parent=successor)
        if nodeIsLeftChild:
            if node.parent is not None:
                nodeParent.setLeft(successor)
        else:
            if node.parent is not None:
                nodeParent.setRight(successor)
        if isRoot:
            self.root = successor
        node.setParent(None)
        node.setLeft(None)
        node.setRight(None)
        if succParent != node:
            return self.updateAfterOp(succParent)
        else:
            return self.updateAfterOp(successor)

    """deletes the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list to be deleted
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def delete(self, i):
        if self.size < (i + 1) or i < 0:
            return -1
        nodeToDelete = self.treeSelect(i + 1)
        if self.size == 1:
            self.root = None
            nodeToDelete.setParent(None)
            nodeToDelete.setRight(None)
            nodeToDelete.setLeft(None)
            self.size = self.size - 1
            self.firstItem = None
            self.lastItem = None
            return 0
        self.size = self.size - 1
        parent = nodeToDelete.getParent()
        isRoot = False
        isLeftChild = False
        if parent is None:
            isRoot = True
        else:
            if nodeToDelete == parent.getLeft():
                isLeftChild = True
            else:
                isLeftChild = False

        if self.lastItem == nodeToDelete:
            self.lastItem = self.predecessor(nodeToDelete)
        if self.firstItem == nodeToDelete:
            self.firstItem = self.successor(nodeToDelete)

        """this node is a leaf"""
        if nodeToDelete.getLeft().isRealNode() is False and nodeToDelete.getRight().isRealNode() is False:
            self.deleteLeaf(nodeToDelete, isRoot, parent, isLeftChild)
            numOfRotations = self.updateAfterOp(parent)
            return numOfRotations

            """this node has only left child"""
        elif nodeToDelete.getLeft().isRealNode() is True and nodeToDelete.getRight().isRealNode() is False:
            self.deleteNodeWithLeftChild(nodeToDelete, isRoot, parent, isLeftChild)
            numOfRotations = self.updateAfterOp(parent)
            return numOfRotations

            """this node has only right child"""
        elif nodeToDelete.getRight().isRealNode() is True and nodeToDelete.getLeft().isRealNode() is False:
            self.deleteNodeWithRightChild(nodeToDelete, isRoot, parent, isLeftChild)
            numOfRotations = self.updateAfterOp(parent)
            return numOfRotations

            """this node has two children"""
        elif nodeToDelete.getRight().isRealNode() is True and nodeToDelete.getLeft().isRealNode() is True:
            return self.deleteNodeWithTwoChildren(nodeToDelete, isRoot, parent, isLeftChild)

    """returns the value of the first item in the list

    @rtype: str
    @returns: the value of the first item, None if the list is empty
    """

    def first(self):
        if self.empty():
            return None
        return self.firstItem.getValue()

    """returns the value of the last item in the list

    @rtype: str
    @returns: the value of the last item, None if the list is empty
    """

    def last(self):
        if self.empty():
            return None
        return self.lastItem.getValue()

    """returns an array representing list 

    @rtype: list
    @returns: a list of strings representing the data structure
    """

    def listToArray(self):
        lst = []
        if self.empty():
            return lst
        self.listToArrayRec(self.root, lst)
        return lst

    def listToArrayRec(self, node, lst):
        if node.isRealNode():
            self.listToArrayRec(node.getLeft(), lst)
            lst.append(node.getValue())
            self.listToArrayRec(node.getRight(), lst)

    """returns the size of the list 

    @rtype: int
    @returns: the size of the list
    """

    def length(self):
        return self.size

    """sort the info values of the list

    @rtype: list
    @returns: an AVLTreeList where the values are sorted by the info of the original list.
    """

    def sort(self):
        sortedTree = AVLTreeList()
        lst = self.listToArray()
        length = len(lst)
        cntNone = 0
        lst1 = []
        for i in range(length):
            if lst[i] is None:
                cntNone += 1
            else:
                lst1.append(lst[i])
        lst1 = self.mergesort(lst1)
        for i in range(cntNone):
            lst1.append(None)
        for i in range(length):
            sortedTree.insert(i, lst1[i])
        return sortedTree

    """merging A and B
    
    @type A: list
    @type B: list
    @pre: A and B must be sorted
    @rtype: list
    @returns: merged list
    """

    def merge(self, A, B):
        n = len(A)
        m = len(B)
        C = [None for i in range(n + m)]

        a = 0
        b = 0
        c = 0
        while a < n and b < m:  # more element in both A and B
            if A[a] < B[b]:
                C[c] = A[a]
                a += 1
            else:
                C[c] = B[b]
                b += 1
            c += 1

        C[c:] = A[a:] + B[b:]  # append remaining elements (one of those is empty)

        return C

    """sorting lst

    @type lst: list
    @rtype: list
    @returns: sorted list
    """

    def mergesort(self, lst):
        n = len(lst)
        if n <= 1:
            return lst
        else:  # two recursive calls, then merge
            return self.merge(self.mergesort(lst[0:n // 2]), self.mergesort(lst[n // 2:n]))

    """permute the info values of the list 

    @rtype: list
    @returns: an AVLTreeList where the values are permuted randomly by the info of the original list. ##Use Randomness
    """

    def permutation(self):
        lst = self.listToArray()
        n = len(lst)
        for i in range(n - 1, 0, -1):  # this part shuffles lst
            j = random.randint(0, i)
            val = lst[i]
            lst[i] = lst[j]
            lst[j] = val
        return self.createAVLTreeFromArray(lst)

    """Envelop function of create AVL Tree From Array
    
    @type lst: list
    @rtype: AVLTreeList
    @returns: an AVLTreeList that create from lst
    """

    def createAVLTreeFromArray(self, lst):
        tree = AVLTreeList()
        n = len(lst)
        if n == 0:
            return tree
        tree.root = self.createAVLTreeFromArrayRec(lst, 0, n - 1)
        tree.size = n
        tree.firstItem = tree.treeSelect(1)
        tree.lastItem = tree.treeSelect(n)
        return tree

    """ create AVL Tree From Array recursive

    @type lst: list
    @type first: int
    @type last: int
    @rtype: AVLTreeList
    @returns: an AVLTreeList that create from positions first to last in lst
    """

    def createAVLTreeFromArrayRec(self, lst, first, last):
        index = first + (last - first) // 2
        if first == last:
            new = AVLNode(lst[first])
            virt1 = AVLNode(None)
            virt2 = AVLNode(None)
            self.insertAsRightSon(son=virt1, parent=new)
            self.insertAsLeftSon(son=virt2, parent=new)
            new.setHeight(max(new.getLeft().getHeight(), new.getRight().getHeight()) + 1)
            new.setSize(new.getLeft().getSize() + new.getRight().getSize() + 1)
            return new

        if first > last:
            return AVLNode(None)

        root = AVLNode(lst[index])
        root.setRight(self.createAVLTreeFromArrayRec(lst, index + 1, last))
        root.getRight().setParent(root)
        root.setLeft(self.createAVLTreeFromArrayRec(lst, first, index - 1))
        root.getLeft().setParent(root)

        root.setHeight(max(root.getLeft().getHeight(), root.getRight().getHeight()) + 1)
        root.setSize(root.getLeft().getSize() + root.getRight().getSize() + 1)
        return root

    """concatenates lst to self

    @type lst: AVLTreeList
    @param lst: a list to be concatenated after self
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    """

    def concat(self, lst):
        if self.root is None and lst.root is None:
            return 0
        if self.root is None:
            self.root = lst.root
            self.size = lst.size
            self.firstItem = lst.firstItem
            self.lastItem = lst.lastItem
            return abs(lst.root.getHeight() - (-1))
        if lst.root is None:
            return abs(self.root.getHeight() - (-1))
        """a is self.root, b is the node in lst such that b.height<= self.height,
                 x is the node such that self<x<lst, c will be the parent of x"""
        heightDiff = abs(self.root.getHeight() - lst.root.getHeight())
        x = self.lastItem
        self.delete(self.size - 1)
        self.insertAsLeftSon(son=AVLNode(None), parent=x)
        self.insertAsRightSon(son=AVLNode(None), parent=x)
        self.size = self.size + lst.size + 1
        if self.root is None:  # that means that original self had only one node
            self.insertAsLeftSon(x, lst.firstItem)
            self.root = lst.root
            self.size = lst.size + 1
            self.firstItem = lst.firstItem
            self.lastItem = lst.lastItem
        elif self.root.getHeight() == lst.root.getHeight():
            self.insertAsLeftSon(son=self.root, parent=x)
            self.insertAsRightSon(son=lst.root, parent=x)
            self.root = x
            self.lastItem = lst.lastItem
            x.setHeight(max(x.getLeft().getHeight(), x.getRight().getHeight()) + 1)
            x.setSize(x.getLeft().getSize() + x.getRight().getSize() + 1)
            return heightDiff
        elif self.root.getHeight() < lst.root.getHeight():
            a = self.root
            b = lst.root
            while b.getHeight() > self.root.getHeight():
                b = b.getLeft()
            c = b.getParent()
            self.insertAsLeftSon(son=x, parent=c)
            self.insertAsRightSon(son=b, parent=x)
            self.insertAsLeftSon(son=a, parent=x)
            self.root = lst.root
        else:
            a = lst.root
            b = self.root
            while b.getHeight() > lst.root.getHeight():
                b = b.getRight()
            c = b.getParent()
            self.insertAsRightSon(son=x, parent=c)
            self.insertAsLeftSon(son=b, parent=x)
            self.insertAsRightSon(son=a, parent=x)
        self.lastItem = lst.lastItem
        self.updateAfterOp(x)
        return heightDiff

    """searches for a *value* in the list

    @type val: str
    @param val: a value to be searched
    @rtype: int
    @returns: the first index that contains val, -1 if not found.
    """

    def search(self, val):
        if self.empty():
            return -1
        lst = self.listToArray()
        n = len(lst)
        for i in range(n):
            if lst[i] == val:
                return i
        return -1

    """returns the root of the tree representing the list

    @rtype: AVLNode
    @returns: the root, None if the list is empty
    """

    def getRoot(self):
        return self.root

    """retrieves the successor of node

      @type node: AVLNode
      @pre: the node is in the tree
      @rtype: AVLNode
      @returns: the successor of node
      """

    def successor(self, node):
        rightChild = node.getRight()
        if rightChild.isRealNode() is True:
            while rightChild.getLeft().isRealNode():
                rightChild = rightChild.getLeft()
            return rightChild
        y = node.getParent()
        while y is not None and node == y.right:
            node = y
            y = node.getParent()
        return y

    """right rotation

       @type node: AVLNode
       @param node: node is an AVL CRIMINAL with balance factor +2. his left son BFS is +1 or 0
       """

    def rightRotation(self, node):
        firstLeftChild = node.getLeft()
        parent = node.getParent()
        flagLeft = False
        if self.root == node:
            self.root = firstLeftChild
        else:
            if parent.getLeft() == node:
                flagLeft = True
        node.setLeft(firstLeftChild.getRight())
        firstLeftChild.getRight().setParent(node)
        firstLeftChild.setRight(node)
        firstLeftChild.setParent(parent)
        if flagLeft and parent is not None:
            parent.setLeft(firstLeftChild)
        elif parent is not None:
            parent.setRight(firstLeftChild)
        node.setParent(firstLeftChild)

    """left rotation

    @type node: AVLNode
    @param node: node is an AVL CRIMINAL with balance factor -2. his left son BFS is -1 or 0
    """

    def leftRotation(self, node):
        firstRightChild = node.getRight()
        flagLeft = False
        parent = node.getParent()
        if self.root == node:
            self.root = firstRightChild
        else:
            if parent.getLeft() == node:
                flagLeft = True
        node.setRight(firstRightChild.getLeft())
        firstRightChild.getLeft().setParent(node)
        firstRightChild.setLeft(node)
        firstRightChild.setParent(parent)
        if flagLeft and parent is not None:
            parent.setLeft(firstRightChild)
        elif parent is not None:
            parent.setRight(firstRightChild)
        node.setParent(firstRightChild)

    """left then right rotation
    @type node: AVLNode
    @param node: node is an AVL CRIMINAL with balance factor +2. his left son BFS is -1 
    """

    def leftThenRightRotation(self, node):
        self.leftRotation(node.getLeft())
        self.rightRotation(node)

    """right then left rotation
    @type node: AVLNode
    @param node: node is an AVL CRIMINAL with balance factor +2. his left son BFS is -1 
    """

    def rightThenLeftRotation(self, node):
        self.rightRotation(node.getRight())
        self.leftRotation(node)
