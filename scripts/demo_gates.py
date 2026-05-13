from gradflow.gates import (
    add_forward,
    add_backward,
    mul_forward,
    mul_backward,
    max_forward,
    max_backward,
    copy_forward,
    copy_backward,
)


def demo_add_gate():
    x = 3
    y = 5
    upstream_grad = 2

    z = add_forward(x, y)
    dx, dy = add_backward(upstream_grad)

    print("Add Gate")
    print("--------")
    print(f"x = {x}, y = {y}")
    print(f"z = x + y = {z}")
    print(f"upstream_grad = {upstream_grad}")
    print(f"dx = {dx}, dy = {dy}")
    print()


def demo_mul_gate():
    x = 3
    y = 5
    upstream_grad = 2

    z = mul_forward(x, y)
    dx, dy = mul_backward(x, y, upstream_grad)

    print("Multiply Gate")
    print("-------------")
    print(f"x = {x}, y = {y}")
    print(f"z = x * y = {z}")
    print(f"upstream_grad = {upstream_grad}")
    print(f"dx = {dx}, dy = {dy}")
    print()


def demo_max_gate():
    x = 3
    y = 5
    upstream_grad = 2

    z = max_forward(x, y)
    dx, dy = max_backward(x, y, upstream_grad)

    print("Max Gate")
    print("--------")
    print(f"x = {x}, y = {y}")
    print(f"z = max(x, y) = {z}")
    print(f"upstream_grad = {upstream_grad}")
    print(f"dx = {dx}, dy = {dy}")
    print()


def demo_copy_gate():
    x = 4
    x1, x2 = copy_forward(x, n=2)

    grad_from_branch_1 = 2
    grad_from_branch_2 = 1

    dx = copy_backward(grad_from_branch_1, grad_from_branch_2)

    print("Copy Gate")
    print("---------")
    print(f"x = {x}")
    print(f"x1 = {x1}, x2 = {x2}")
    print(f"grad_from_branch_1 = {grad_from_branch_1}")
    print(f"grad_from_branch_2 = {grad_from_branch_2}")
    print(f"dx = {dx}")
    print()


if __name__ == "__main__":
    demo_add_gate()
    demo_mul_gate()
    demo_max_gate()
    demo_copy_gate()
