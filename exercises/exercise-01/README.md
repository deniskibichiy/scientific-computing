# Exercise 01: NumPy Arrays and Numerical Computation

## 1. Mathematical Concept

An array is an organized collection of numbers. A vector is a one-dimensional array, while a matrix is a rectangular array of numbers.

Scientific computing frequently requires the same mathematical operation to be applied to many values. NumPy provides arrays and vectorized operations that allow these calculations to be performed directly on the entire array without explicitly writing a loop for each element.

For example:

```text
x = [1, 2, 3]

x² = [1, 4, 9]
```

NumPy can perform this operation directly on an array.

## 2. Basic Example

Given the temperatures:

```text
[20, 22, 24, 26] °C
```

the mean temperature is:

```text
(20 + 22 + 24 + 26) / 4 = 23 °C
```

NumPy provides the `np.mean()` function to perform this calculation directly.

## 3. Real-World Applications

NumPy arrays are useful in:

1. IoT sensor readings
2. Image pixels
3. Network traffic measurements
4. Scientific measurements
5. Machine-learning feature matrices
6. Financial time-series data

## 4. Real-World Problem

A weather station records 24 hourly temperatures in degrees Celsius.

The system needs to:

* Store the readings in a NumPy array.
* Calculate the daily mean temperature.
* Find the minimum temperature.
* Find the maximum temperature.
* Convert all temperatures from Celsius to Fahrenheit.

The Celsius-to-Fahrenheit formula is:

```text
F = 1.8C + 32
```

## 5. Solution Approach

The 24 temperature readings are represented using a single NumPy array.

NumPy's:

* `np.mean()` calculates the average.
* `np.min()` finds the minimum value.
* `np.max()` finds the maximum value.

The Celsius-to-Fahrenheit conversion is applied to the entire array using a vectorized NumPy expression:

```text
temperatures_f = 1.8 * temperatures_c + 32
```

This avoids manually processing each temperature with a Python loop.

## 6. Implementation

The implementation is provided in:

`Task_02_numpy_arrays.py`

The program calculates the daily temperature statistics and performs the Celsius-to-Fahrenheit conversion.

## 7. Expected Output

The program should display:

* Daily mean temperature
* Minimum temperature
* Maximum temperature
* The first six Fahrenheit readings
