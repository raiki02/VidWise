package qdrant

import (
	"context"
	"fmt"
	"log/slog"

	pb "github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type Client struct {
	Conn        *grpc.ClientConn
	Points      pb.PointsClient
	Collections pb.CollectionsClient
}

func NewClient(ctx context.Context, addr string) (*Client, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("qdrant dial: %w", err)
	}

	points := pb.NewPointsClient(conn)
	collections := pb.NewCollectionsClient(conn)

	// Health check: list collections
	_, err = collections.List(ctx, &pb.ListCollectionsRequest{})
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("qdrant health check: %w", err)
	}
	slog.Info("qdrant.connected", "addr", addr)
	return &Client{Conn: conn, Points: points, Collections: collections}, nil
}

func (c *Client) Close() error {
	return c.Conn.Close()
}
