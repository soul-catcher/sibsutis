(defun searc (item ls &optional (ind 1))
    (if ls
        (if(=(car ls)item)
            (cons ind (searc item(cdr ls)(+ ind 1)))
            (searc item(cdr ls)(+ ind 1))
        )
    )
)

(princ(searc '4 '(8 3 4 6 4 1)))
