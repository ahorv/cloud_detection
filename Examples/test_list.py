
class processing_2(object):

    def __init__(self):
        self.delta = 10
        self.slideshow_step = 0
        self.openCVImg_initial_img_list = []
        self.pixMapImg_final_img_llist = []

        for elem in range(101):
            self.pixMapImg_final_img_llist.append('X')

        length = len(self.pixMapImg_final_img_llist)
        firstelem = self.pixMapImg_final_img_llist[0]
        lastelem = self.pixMapImg_final_img_llist[-1]
        print('#------------------------------------------------------------#')
        print('#Final:   Length: {} | First Elem: {} | Last Elem: {} '.format(length,firstelem,lastelem))


        for elem in range(101):
            self.openCVImg_initial_img_list.append(elem)

        length = len(self.openCVImg_initial_img_list)
        firstelem = self.openCVImg_initial_img_list[0]
        lastelem = self.openCVImg_initial_img_list[-1]

        print('#OpenCV:  Length: {} | First Elem: {} | Last Elem: {} '.format(length,firstelem,lastelem))
        print('#------------------------------------------------------------#')

        self.tot_numb_of_images = len(self.openCVImg_initial_img_list)

    def doImageProcessing(self, list):
        try:
            whereException = ''
            cur_idx = self.slideshow_step
            last_idx = self.tot_numb_of_images
            delta = self.delta

            if cur_idx + delta > last_idx:
                start_1 = (cur_idx + delta) - last_idx
                start_2 = 0
                stop_1 = last_idx
                stop_2 = start_1
                print('------------------------------------------------------------------------------')
                print('I   | delta: {} | idx: {}  -> start_1: {} | stop_1: {} start_2: {} | stop_2: {}'.format(delta,cur_idx,start_1,stop_1,start_2,stop_2))
                print('------------------------------------------------------------------------------')
                whereException = 'I | '

            elif cur_idx + delta < last_idx:
                start_1 = (cur_idx + delta)
                start_2 = 0
                stop_1 = last_idx
                stop_2 = start_1
                print('-------------------------------------------------------------------------------')
                print(' II   | delta: {} | idx: {} -> start_1: {} | stop_1: {} start_2: {} | stop_2: {}'.format(delta,cur_idx,start_1,start_2,stop_2))
                print('-------------------------------------------------------------------------------')
                whereException = 'II | '

            elif cur_idx + delta == last_idx:
                start_1 = 0
                print('-----------------------------------------------------------------------')
                print(' III | delta: {} | idx: {} -> start_1: {} | stop_2: {}'.format(delta, cur_idx, start_1, last_idx))
                print('-----------------------------------------------------------------------')

                for index in range(last_idx):

                    if index == 0:
                        idx_prev = last_idx - 1
                        idx_next = index
                    else:
                        idx_prev = index - 1
                        idx_next = index

                    whereException = 'I '
                    prev = self.openCVImg_initial_img_list[idx_prev]
                    next = self.openCVImg_initial_img_list[idx_next]
                    list[idx_prev] = prev
                    list[idx_next] = next

                print(', '.join(map(str, list)))
                return

            for index in range(start_1, stop_1):

                if index == 0:
                    idx_prev = last_idx -1
                    idx_next = index
                else:
                    idx_prev = index - 1
                    idx_next = index

                whereException = 'II '
                prev = self.openCVImg_initial_img_list[idx_prev]
                next = self.openCVImg_initial_img_list[idx_next]
                list[idx_prev] = prev
                list[idx_next] = next
            print('For 1: ' + ', '.join(map(str, list)))

            for index in range(start_2, stop_2):
                if index == 0:
                    idx_prev = last_idx -1
                    idx_next = index
                else:
                    idx_prev = index - 1
                    idx_next = index

                whereException = 'III '
                prev = self.openCVImg_initial_img_list[idx_prev]
                next = self.openCVImg_initial_img_list[idx_next]
                list[idx_prev] = prev
                list[idx_next] = next
            print('For 2: ' + ', '.join(map(str, list)))


        except Exception as e:
            print('Where : {} idx: {} idx_prev: {} idx_next: {} -> {}'.format(whereException, index, idx_prev, idx_next, str(e)))

    def doAllTheMagic(self):
        self.doImageProcessing(self.pixMapImg_final_img_llist)

    def setNextImg(self,value):
        self.slideshow_step = value

    def setDelta(self,value):
        self.delta = value


class Processing_1(object):

    def __init__(self):
        self.delta = 10
        self.slideshow_step = 0
        self.openCVImg_initial_img_list = []
        self.pixMapImg_final_img_llist = []

        for elem in range(101):
            self.pixMapImg_final_img_llist.append('X')

        length = len(self.pixMapImg_final_img_llist)
        firstelem = self.pixMapImg_final_img_llist[0]
        lastelem = self.pixMapImg_final_img_llist[-1]
        print('#------------------------------------------------------------#')
        print('#Final:   Length: {} | First Elem: {} | Last Elem: {} '.format(length,firstelem,lastelem))


        for elem in range(101):
            self.openCVImg_initial_img_list.append(elem)

        length = len(self.openCVImg_initial_img_list)
        firstelem = self.openCVImg_initial_img_list[0]
        lastelem = self.openCVImg_initial_img_list[-1]

        print('#OpenCV:  Length: {} | First Elem: {} | Last Elem: {} '.format(length,firstelem,lastelem))
        print('#------------------------------------------------------------#')

        self.tot_numb_of_images = len(self.openCVImg_initial_img_list)

    def doImageProcessing(self, list):
       try:

            whereException = ''
            cur_idx  = self.slideshow_step
            last_idx = self.tot_numb_of_images
            delta = self.delta

            if cur_idx + delta > last_idx:
                start_1 = (cur_idx + delta) - last_idx
                start_2 = 0
                stop_1 = last_idx
                stop_2 = start_1
                print('------------------------------------------------------------------------------')
                print('I   | delta: {} | idx: {}  -> start_1: {} | stop_1: {} start_2: {} | stop_2: {}'.format(delta,cur_idx,start_1,stop_1,start_2,stop_2))
                print('------------------------------------------------------------------------------')
                whereException = 'I | '

            elif cur_idx + delta < last_idx:
                start_1 = (cur_idx + delta)
                start_2 = 0
                stop_1 = last_idx
                stop_2 = start_1
                print('-------------------------------------------------------------------------------')
                print('II   | delta: {} | idx: {} -> start_1: {} | stop_1: {} start_2: {} | stop_2: {}'.format(delta,cur_idx, start_1, stop_1,start_2, stop_2))
                print('-------------------------------------------------------------------------------')
                whereException = 'II | '

            elif cur_idx + delta == last_idx:
                start_1 = 0
                print('-----------------------------------------------------------------------')
                print('III | delta: {} | idx: {} -> start_1: {} | stop_2: {}'.format(delta,cur_idx,start_1,last_idx))
                print('-----------------------------------------------------------------------')

                cnt = 0

                for index in range(last_idx):
                    whereException = '(1): [{}] |  at idx: {} '.format(cnt,index)
                    cnt += 1
                    image = self.openCVImg_initial_img_list[index]
                    list[index] = image
                    #print('(1): [{}]  | Start: {} - Stop: {} | idx: {} |-> {}'.format(cnt, 0, last_idx, index, image))
                print(', '.join(map(str, list)))
                return

            cnt = 0

            for index in range(start_1,stop_1):
                whereException = '(2): [{}] |  at idx: {} '.format(cnt,index)
                cnt +=1
                image = self.openCVImg_initial_img_list[index]
                list[index] = image
                #print('(2): [{}]  | Start: {} - Stop: {} | idx: {} |-> {}'.format(cnt, start_1, stop_1, index,image))
            print('For 1: '+', '.join(map(str, list)))

            cnt = 0

            for index in range(start_2, stop_2):
                whereException = '(3): [{}] |  at idx: {} '.format(cnt,index)
                cnt += 1
                image = self.openCVImg_initial_img_list[index]
                list[index] = image
                #print('(3): [{}]  | Start: {} - Stop: {} | idx: {} |-> {}'.format(cnt, start_2, stop_2, index, image))
            print('For 2: '+', '.join(map(str, list)))

       except Exception as e:
            print('Where : {}{}'.format(whereException , str(e)))

    def doAllTheMagic(self):
        self.doImageProcessing(self.pixMapImg_final_img_llist)

    def setNextImg(self,value):
        self.slideshow_step = value

    def setDelta(self,value):
        self.delta = value

def main():
    try:
        #proc_1 = Processing_1()
        #proc_1.setNextImg(92)
        #proc_1.setDelta(10)
        #proc_1.doAllTheMagic()

        proc_2 = processing_2()
        proc_2.setNextImg(92)
        proc_2.setDelta(10)
        proc_2.doAllTheMagic()

        '''
        for index, item in enumerate(proc.getFinalList(), start=0):
            print('idx:{} | item: {} '.format(index, item))
        '''

    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()