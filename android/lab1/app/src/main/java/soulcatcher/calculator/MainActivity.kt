package soulcatcher.calculator

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import kotlinx.android.synthetic.main.activity_main.*
import kotlin.text.StringBuilder

class MainActivity : AppCompatActivity() {

    private var arg = StringBuilder()
    private var res = -1
    private var flag = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val resView = findViewById<TextView>(R.id.resView)
    }

    fun onClickNum(view: View) {
        val btn = findViewById<Button>(R.id.button1)
        val btn_2 = findViewById<Button>(R.id.button2)
        val btn_3 = findViewById<Button>(R.id.button3)
        val btn_5 = findViewById<Button>(R.id.button4)
        val btn_6 = findViewById<Button>(R.id.button5)
        val btn_7 = findViewById<Button>(R.id.button6)
        val btn_9 = findViewById<Button>(R.id.button7)
        val btn_10 = findViewById<Button>(R.id.button8)
        val btn_11 = findViewById<Button>(R.id.button9)
        val btn_13 = findViewById<Button>(R.id.button0)
        when (view) {
            btn -> arg.append(btn.text.toString())
            btn_2 -> arg.append(btn_2.text.toString())
            btn_3 -> arg.append(btn_3.text.toString())
            btn_10 -> arg.append(btn_10.text.toString())
            btn_5 -> arg.append(btn_5.text.toString())
            btn_6 -> arg.append(btn_6.text.toString())
            btn_7 -> arg.append(btn_7.text.toString())
            btn_9 -> arg.append(btn_9.text.toString())
            btn_11 -> arg.append(btn_11.text.toString())
            btn_13 -> arg.append(btn_13.text.toString())
        }
        resView.text = arg.toString()
    }

    fun onClickAction(view: View) {
        val btn_sum = findViewById<Button>(R.id.buttonPlus)
        val btn_subtract = findViewById<Button>(R.id.buttonMinus)
        val btn_mult = findViewById<Button>(R.id.buttonMultiply)
        val btn_div = findViewById<Button>(R.id.buttonDivide)
        val btn_res = findViewById<Button>(R.id.buttonRes)
        val btn_del = findViewById<Button>(R.id.buttonClear)
        when (view) {
            btn_div -> {
                if (arg.isNotEmpty()) {
                    flag = 1
                    res = arg.toString().toInt()
                    arg.clear()
                } else {
                    flag = 1
                }
            }
            btn_sum -> {
                if (arg.isNotEmpty()) {
                    flag = 2
                    res = arg.toString().toInt()
                    arg.clear()
                } else {
                    flag = 2
                }
            }
            btn_subtract -> {
                if (arg.isNotEmpty()) {
                    flag = 3
                    res = arg.toString().toInt()
                    arg.clear()
                } else {
                    flag = 3
                }
            }
            btn_mult -> {
                if (arg.isNotEmpty()) {
                    flag = 4
                    res = arg.toString().toInt()
                    arg.clear()
                } else {
                    flag = 4
                }
            }
            btn_del -> {
                if (arg.isNotEmpty()) {
                    arg.deleteCharAt(arg.length - 1)
                    resView.text = arg.toString()
                } else if (flag == 5) {
                    resView.text = arg.toString()
                }
            }
            btn_res -> {
                when(flag) {
                    1 -> {
                        if (arg.isNotEmpty()) {
                            if (arg.toString() == "0") {
                                resView.text = String.format("%s", getString(R.string.divby0))
                                arg.clear()
                            } else {
                                res /= arg.toString().toInt()
                                arg.clear()
                                resView.text = String.format("%d", res)
                            }
                        }
                        flag = 5
                    }
                    2 -> {
                        if (arg.isNotEmpty()) {
                            res += arg.toString().toInt()
                            arg.clear()
                            resView.text = String.format("%d", res)
                        }
                        flag = 5
                    }
                    3 -> {
                        if (arg.isNotEmpty()) {
                            res -= arg.toString().toInt()
                            arg.clear()
                            resView.text = String.format("%d", res)
                        }
                        flag = 5
                    }
                    4 -> {
                        if (arg.isNotEmpty()) {
                            res *= arg.toString().toInt()
                            arg.clear()
                            resView.text = String.format("%d", res)
                        }
                        flag = 5
                    }
                }
            }
        }
    }
}
