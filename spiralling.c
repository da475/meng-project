
/*
 * Author : Jaydev Kshirsagar
 *
 * The program in this file emits a re-mapping for pixel coordinates of an image
 * such that the new image will have the pixels rotated in a spiral fashion
 *
 */


#include <stdio.h>
#include <math.h>


static int N = 6;


static int MAX (int X, int Y)
{
	if (X > Y)
		return X;
	else
		return Y;
}


static int MapWithinN (int X)
{
	return (((X - 1) % N) + 1);
}


static int ManhattanDistance (int X1, int Y1, int X2, int Y2)
{
	return (abs (X2 - X1) + abs (Y2 - Y1));
}


static int CityBlockDistance (int X1, int Y1, int X2, int Y2)
{
	return (MAX (abs (X2 - X1), abs (Y2 - Y1)));
}


static int LabelAtLocation (int x, int y)
{
	int ring, Nby2, PositionInRing, RingSize, InteriorSize;

	Nby2 = N >> 1;

	ring = CityBlockDistance(1 + Nby2, Nby2, x, y) + ((x > y) ? 1 : 0);

	RingSize = 4 * ((2 * ring) - 1);

	PositionInRing = ManhattanDistance(Nby2 - ring + 1, Nby2 - ring + 1, x, y);

	if (y > x)
	{
		PositionInRing = RingSize - PositionInRing;
	}

	PositionInRing--;

	if (PositionInRing < 0)
	{
		PositionInRing = RingSize - 1;
	}

	InteriorSize = 4 * (ring - 1) * (ring - 1);

	return PositionInRing + InteriorSize;
}


int main(int argc, char *argv[])
{
	int i, j;

	//printf ("\n");

	if (argc < 2)
	{
		printf("\n Input the size ! \n");
		return -1;
	}

	N = atoi(argv[1]);

	if (N % 2)
	{
		printf("\n Size should be Even ! \n");
		return - 1;
	}

	for (i = 1; i <= N; i++)
	{
		for (j = 1; j <= N; j++)
		{
			printf("%d\n", LabelAtLocation(i, j));
		}

		//printf ("\n");
	}

	//printf ("\n");

	return 0;
}
