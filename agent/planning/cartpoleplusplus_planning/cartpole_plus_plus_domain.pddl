(define (domain cartpole-plus-plus)

  (:requirements :typing :fluents :time :negative-preconditions)

  (:types dummy block)

  (:predicates
    (total_failure) (ready) (cart_available)
  )

  (:functions
    (pos_x) (pos_y) (pos_x_dot) (pos_x_ddot) (pos_y_dot) (pos_y_ddot)
    (m_cart) (friction_cart)
    (theta_x) (theta_y) (theta_x_dot) (theta_y_dot) (theta_x_ddot) (theta_y_ddot)
    (l_pole) (m_pole) (friction_pole)
    (gravity) (F_x) (F_y) (elapsed_time) (inertia)
    (time_limit) (angle_limit) (force_mag)
    (wall_x_min) (wall_x_max) (wall_y_min) (wall_y_max) (wall_z_min) (wall_z_max)
    (block_x ?bl - block) (block_y ?bl - block) (block_z ?bl - block) (block_r ?bl - block)
    (block_x_dot ?bl - block) (block_y_dot ?bl - block) (block_z_dot ?bl - block)
  )


  ; TEMP =  	(/ (+ (F) (* (* (m_pole) (l_pole)) (* (* (theta_dot) (theta_dot)) (SIN_THETA) ) ) ) (+ (m_cart) (m_pole)) )
  ; COS_THETA = (/ (- 32400 (* 4 (* (* (theta) 57.295779513) (* (theta) 57.295779513)))) (+ 32400 (* (* (theta) 57.295779513) (* (theta) 57.295779513))))
  ; SIN_THETA = (/ (* (* 4 (* (theta) 57.295779513)) (- 180 (* (theta) 57.295779513))) (- 40500 (* (* (theta) 57.295779513) (- 180 (* (theta) 57.295779513)))))
  ; upd_COS_THETA = ((-5x^2)/(32400+x^2)) + 1     =        (+ (/ (* (- 0.0 5.0) (* (* (theta) 57.295779513) (* (theta) 57.295779513)) ) (+ 32400 (* (* (theta) 57.295779513) (* (theta) 57.295779513) )) ) 1.0)


  (:event set_force
      :parameters (?d - dummy)
      :precondition (and (ready) (not (total_failure)))
      :effect (and
      	(cart_available)

	      (assign (pos_x_ddot)
          (-
            (/ (+ (F_x) (* (* (m_pole) (l_pole)) (* (* (theta_x_dot) (theta_x_dot)) (/ (* (* 4 (* (theta_x) 57.295779513)) (- 180 (* (theta_x) 57.295779513))) (- 40500 (* (* (theta_x) 57.295779513) (- 180 (* (theta_x) 57.295779513))))) ) ) ) (+ (m_cart) (m_pole)) )
            (/ (* (* (m_pole) (l_pole)) (* (theta_x_ddot) (+ (/ (* (- 0.0 5.0) (* (* (theta_x) 57.295779513) (* (theta_x) 57.295779513)) ) (+ 32400 (* (* (theta_x) 57.295779513) (* (theta_x) 57.295779513) )) ) 1.0) )) (+ (m_cart) (m_pole)) )
        )
        )

        (assign (theta_x_ddot)
      		(/
      			(- (* (gravity) (/ (* (* 4 (* (theta_x) 57.295779513)) (- 180 (* (theta_x) 57.295779513))) (- 40500 (* (* (theta_x) 57.295779513) (- 180 (* (theta_x) 57.295779513))))) ) (* (+ (/ (* (- 0.0 5.0) (* (* (theta_x) 57.295779513) (* (theta_x) 57.295779513)) ) (+ 32400 (* (* (theta_x) 57.295779513) (* (theta_x) 57.295779513) )) ) 1.0) (/ (+ (F_x) (* (* (m_pole) (l_pole)) (* (* (theta_x_dot) (theta_x_dot)) (/ (* (* 4 (* (theta_x) 57.295779513)) (- 180 (* (theta_x) 57.295779513))) (- 40500 (* (* (theta_x) 57.295779513) (- 180 (* (theta_x) 57.295779513))))) ) ) ) (+ (m_cart) (m_pole)) ) ) )
      			(* (l_pole) (- (/ 4.0 3.0) (/ (* (m_pole) (* (+ (/ (* (- 0.0 5.0) (* (* (theta_x) 57.295779513) (* (theta_x) 57.295779513)) ) (+ 32400 (* (* (theta_x) 57.295779513) (* (theta_x) 57.295779513) )) ) 1.0) (+ (/ (* (- 0.0 5.0) (* (* (theta_x) 57.295779513) (* (theta_x) 57.295779513)) ) (+ 32400 (* (* (theta_x) 57.295779513) (* (theta_x) 57.295779513) )) ) 1.0) )) (+ (m_cart) (m_pole)) ) ) )
  			)
      	)

        (assign (pos_y_ddot)
          (-
            (/ (+ (F_y) (* (* (m_pole) (l_pole)) (* (* (theta_y_dot) (theta_y_dot)) (/ (* (* 4 (* (theta_y) 57.295779513)) (- 180 (* (theta_y) 57.295779513))) (- 40500 (* (* (theta_y) 57.295779513) (- 180 (* (theta_y) 57.295779513))))) ) ) ) (+ (m_cart) (m_pole)) )
            (/ (* (* (m_pole) (l_pole)) (* (theta_y_ddot) (+ (/ (* (- 0.0 5.0) (* (* (theta_y) 57.295779513) (* (theta_y) 57.295779513)) ) (+ 32400 (* (* (theta_y) 57.295779513) (* (theta_y) 57.295779513) )) ) 1.0) )) (+ (m_cart) (m_pole)) )
        )
        )

        (assign (theta_y_ddot)
      		(/
      			(- (* (gravity) (/ (* (* 4 (* (theta_y) 57.295779513)) (- 180 (* (theta_y) 57.295779513))) (- 40500 (* (* (theta_y) 57.295779513) (- 180 (* (theta_y) 57.295779513))))) ) (* (+ (/ (* (- 0.0 5.0) (* (* (theta_y) 57.295779513) (* (theta_y) 57.295779513)) ) (+ 32400 (* (* (theta_y) 57.295779513) (* (theta_y) 57.295779513) )) ) 1.0) (/ (+ (F_y) (* (* (m_pole) (l_pole)) (* (* (theta_y_dot) (theta_y_dot)) (/ (* (* 4 (* (theta_y) 57.295779513)) (- 180 (* (theta_y) 57.295779513))) (- 40500 (* (* (theta_y) 57.295779513) (- 180 (* (theta_y) 57.295779513))))) ) ) ) (+ (m_cart) (m_pole)) ) ) )
      			(* (l_pole) (- (/ 4.0 3.0) (/ (* (m_pole) (* (+ (/ (* (- 0.0 5.0) (* (* (theta_y) 57.295779513) (* (theta_y) 57.295779513)) ) (+ 32400 (* (* (theta_y) 57.295779513) (* (theta_y) 57.295779513) )) ) 1.0) (+ (/ (* (- 0.0 5.0) (* (* (theta_y) 57.295779513) (* (theta_y) 57.295779513)) ) (+ 32400 (* (* (theta_y) 57.295779513) (* (theta_y) 57.295779513) )) ) 1.0) )) (+ (m_cart) (m_pole)) ) ) )
  			)
      	)

      )
  )

  (:process movement
    :parameters (?d - dummy)
    :precondition (and (ready) (not (total_failure)))
    :effect (and
        (increase (pos_x) (* #t (pos_x_dot)) )
        (increase (pos_y) (* #t (pos_y_dot)) )
        (decrease (pos_x_dot) (* #t (pos_x_ddot)) )
        (increase (pos_y_dot) (* #t (pos_y_ddot)) )
        (decrease (theta_x_dot) (* #t (theta_x_ddot)) )
        (decrease (theta_y_dot) (* #t (theta_y_ddot)) )
        (increase (theta_x) (* #t (theta_x_dot)))
        (increase (theta_y) (* #t (theta_y_dot)))
        (increase (elapsed_time) (* #t 1) )
    )
  )

  (:process block_movement
    :parameters (?bl - block)
    :precondition (and (ready) (not (total_failure)))
    :effect (and
        (increase (block_x ?bl) (* #t (block_x_dot ?bl)) )
        (increase (block_y ?bl) (* #t (block_y_dot ?bl)) )
        (increase (block_z ?bl) (* #t (block_z_dot ?bl)) )
    )
  )

  (:event bounce_block_x
      :parameters (?bl - block)
      :precondition (and (ready) (not (total_failure))
                         (or (>= (block_x ?bl) (wall_x_max))
                             (<= (block_x ?bl) (wall_x_min))
                         )
      )
      :effect (and
          (assign (block_x_dot ?bl) (* (block_x_dot ?bl) -1.0))
      )
  )

  (:event bounce_block_y
      :parameters (?bl - block)
      :precondition (and (ready) (not (total_failure))
                         (or (>= (block_y ?bl) (wall_y_max))
                             (<= (block_y ?bl) (wall_y_min))
                         )
      )
      :effect (and
          (assign (block_y_dot ?bl) (* (block_y_dot ?bl) -1.0))
      )
  )

  (:event bounce_block_z
      :parameters (?bl - block)
      :precondition (and (ready) (not (total_failure))
                         (or (>= (block_z ?bl) (wall_z_max))
                             (<= (block_z ?bl) (wall_z_min))
                         )
      )
      :effect (and
          (assign (block_z_dot ?bl) (* (block_z_dot ?bl) -1.0))
      )
  )

  (:action do_nothing
    :parameters (?d - dummy)
    :precondition (and
      (ready)
      (<= (theta_x) 2) (>= (theta_x) -2) (<= (theta_y) 2) (>= (theta_y) -2)
      (or (> (F_x) 0.0) (< (F_x) 0.0) (> (F_y) 0.0) (< (F_y) 0.0) )
      (cart_available)
      (not (total_failure))
  )
    :effect (and
      (assign (F_y) 0.0)
      (assign (F_x) 0.0)
      (not (cart_available))
    )
  )

  (:action move_cart_right
    :parameters (?d - dummy)
    :precondition (and
    	(ready)
    	(>= (F_x) 0.0)
    	(cart_available)
    	(not (total_failure))
	)
    :effect (and
      (assign (F_x) (* (force_mag) -1.0))
      (assign (F_y) 0.0)
      (not (cart_available))
  	)
  )

  (:action move_cart_left
    :parameters (?d - dummy)
    :precondition (and
    	(ready)
    	(<= (F_x) 0.0)
    	(cart_available)
    	(not (total_failure))
	)
    :effect (and
      (assign (F_x) (force_mag))
      (assign (F_y) 0.0)
      (not (cart_available))
  	)
  )

  (:action move_cart_backward
    :parameters (?d - dummy)
    :precondition (and
      (ready)
      (>= (F_y) 0.0)
      (cart_available)
      (not (total_failure))
  )
    :effect (and
      (assign (F_y) (* (force_mag) -1.0))
      (assign (F_x) 0.0)
      (not (cart_available))
    )
  )

  (:action move_cart_forward
    :parameters (?d - dummy)
    :precondition (and
      (ready)
      (<= (F_y) 0.0)
      (cart_available)
      (not (total_failure))
  )
    :effect (and
      (assign (F_y) (force_mag))
      (assign (F_x) 0.0)
      (not (cart_available))
    )
  )

  (:event exited_goal_region
      :parameters (?d - dummy)
      :precondition (and
          (or (>= (theta_x) (angle_limit))
          (<= (theta_x) (- 0.0 (angle_limit)))
          (>= (theta_y) (angle_limit))
          (<= (theta_y) (- 0.0 (angle_limit)))
          )
          ; (pole_position)
          (not (total_failure))
      )
      :effect (and
      	  ; (not (pole_position))
          (total_failure)
      )
  )

  (:event cart_out_of_bounds
      :parameters (?d - dummy)
      :precondition (and
          (or (>= (pos_x) (wall_x_max))
          (<= (pos_x) (wall_x_min))
          (>= (pos_y) (wall_y_max))
          (<= (pos_y) (wall_y_min))
          )
          (not (total_failure))
      )
      :effect (and
          (total_failure)
      )
  )

  (:event time_limit_reached
      :parameters (?d - dummy)
      :precondition (and
          (> (elapsed_time) (time_limit))
          (not (total_failure))
      )
      :effect (and
          (total_failure)
      )
  )


)
