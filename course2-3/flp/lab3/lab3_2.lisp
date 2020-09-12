(defun f (matr)
    (if matr
        (cons
            (caar matr)
            (f (cdr matr))
        )
    )
)

(defun g (matr)
    (if (car matr)
        (cons
            (cdar matr)
            (g(cdr matr))
        )
    )
)

(defun h (matr)
    (if (car matr)
        (cons (f matr) (h (g matr)))
    )
)

(princ (h '((1 2 3 4) (5 6 7 8) (9 10 11 12))))
