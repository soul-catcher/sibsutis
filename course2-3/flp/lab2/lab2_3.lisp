(defun inf-to-pref(l)
    (eval(cond 
        ((null l) nil)
        ((atom l) l)
        (t (list (cadr l) (inf-to-pref (car l))  (inf-to-pref (caddr l))))
    ))
)

(princ(inf-to-pref '((2 + 6) / (2 + 2))))
