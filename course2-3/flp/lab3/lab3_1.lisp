(defun f (a b)
    (if a
        (if (member(car a) b)
            T
            (f (cdr a) b)
        )
        nil
    )
)

(princ (f '(1 2 3 4) '(7 8 0 4)))
