import mysql.connector
import numpy as np
from sklearn.preprocessing import MinMaxScaler


import tensorflow as tf
from flask import Flask,json,jsonify,make_response,render_template
import pandas as pd

app= Flask(__name__)


@app.route("/model_V1")
def fetch_data():
    mydb = mysql.connector.connect(
        host="sql12.freemysqlhosting.net",
        user="sql12263093",
        password="XaFgzDDEx5",
        database="sql12263093", port = '3306'
    )
    cursor = mydb.cursor()
    cursor.execute("SELECT *  FROM input_data WHERE batchID = (SELECT batch_id FROM batch)")

    result = cursor.fetchall()
    js=json.dumps(result)
    predict = list(result)
    listn = np.array(predict)
    listn = listn[:, 3:32]
    listn = listn.astype(np.float64)

    scaler = MinMaxScaler(copy=False)
    scaler.fit_transform(listn[:, 0:1])
    scaler.fit_transform(listn[:, 4:5])
    scaler.fit_transform(listn[:, 11:23])

    sess = tf.Session()
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('CC_Default_Trained_Model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("X:0")
    feed_dict = {w1: listn}
    op_to_restore = graph.get_tensor_by_name("predictions:0")
    result = sess.run(op_to_restore, feed_dict)
    cursor2 = mydb.cursor()
    cursor3 = mydb.cursor()
    cursor4 = mydb.cursor()
    i = 0

    cursor2.execute("SELECT * FROM batch")
    for row in cursor2.fetchall():
        batch = row
        batch = list(batch)
        batch = np.array(batch)
        batch = batch[0]
        sql = "SELECT cust_name, cust_id FROM input_data WHERE batchID = %s "
        bat = (int(batch),)
        cursor3.execute(sql, bat)

        for row2 in cursor3.fetchall():
            cust_nm, custID = row2
            d_prob = result[i][0]
            nd_prob = result[i][1]
            i = i + 1

            sql2 = "INSERT INTO output_data (batchID, cust_name, cust_id, def_prob, ndef_prob) VALUES (%s,%s,%s,%s,%s)"
            out = (int(batch), cust_nm, custID, float(d_prob), float(nd_prob))
            cursor4.execute(sql2, out)

    mydb.commit()

    return 'everything ok'





if __name__ == '__main__':
    app.run(debug=True)