import pre_train_demo

success = 0
fail = 0
while True:
    print "success :", success
    print "fail:", fail
    try:
        pre_train_demo.mykey()
        success += 1
    except Exception as e:
        print e.message
        fail += 1

